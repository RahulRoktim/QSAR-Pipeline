import time
import pandas as pd

from chembl_webresource_client.new_client import new_client

import config
from src.target_resolver import (
    ResolvedTarget,
    TargetResolutionError,
    choose_best_target,
    verify_configured_target,
)


class ChEMBLDownloader:

    def __init__(self):

        self.target_api = new_client.target
        self.activity_api = new_client.activity

    def resolve_target(self) -> ResolvedTarget:
        """Resolve and verify the ChEMBL target before any data is downloaded.

        A configured TARGET_CHEMBL_ID is not trusted on sight: it is fetched and
        checked against config.TARGET, so a label/id mismatch fails loudly here
        instead of silently mislabelling the whole dataset.
        """
        label = config.TARGET
        configured = config.TARGET_CHEMBL_ID or config.KNOWN_TARGET_IDS.get(label.upper())

        if configured:
            print(f"\nVerifying configured target {configured} against '{label}'...")

            record = self._fetch_target(configured)

            if record is None:
                raise TargetResolutionError(
                    f"ChEMBL returned no record for {configured!r}."
                )

            return verify_configured_target(label, configured, record)

        print(f"\nSearching ChEMBL for '{label}'...\n")

        targets = self._search(label)

        resolved = choose_best_target(label, targets, self.activity_count)

        print(
            f"\nAuto-selected: {resolved.chembl_id} | {resolved.pref_name} "
            "(most activities among matching single-protein targets)"
        )

        return resolved

    # Backwards-compatible shim for callers that only need the identifier.
    def search_target(self):
        return self.resolve_target().chembl_id

    def _fetch_target(self, target_chembl_id):
        for attempt in range(1, 6):
            try:
                results = list(
                    self.target_api.filter(target_chembl_id=target_chembl_id)
                )
                return results[0] if results else None
            except Exception as error:
                print(
                    f"ChEMBL target lookup failed (attempt {attempt}/5): {error}"
                )
                time.sleep(10)

        raise TargetResolutionError(
            "Could not reach ChEMBL after several attempts. The EBI service may "
            "be unavailable - try again later."
        )

    def _search(self, label):
        for attempt in range(1, 6):
            try:
                return list(self.target_api.search(label))
            except Exception as error:
                print(f"ChEMBL target search failed (attempt {attempt}/5): {error}")
                print("The ChEMBL/EBI service may be temporarily unavailable.")
                time.sleep(10)

        raise TargetResolutionError(
            "Could not reach ChEMBL after several attempts. Set "
            "TARGET_CHEMBL_ID to skip the search, or retry later."
        )

    def activity_count(self, target_id):
        try:
            return len(
                self.activity_api.filter(
                    target_chembl_id=target_id,
                    standard_type=config.ACTIVITY_TYPE,
                    pchembl_value__isnull=False,
                )
            )
        except Exception:
            return 0

    def download_pages(self, target_chembl):

        # Download ALL activities that carry a pChEMBL value (not a capped 1000).
        # The previous cap + downstream de-duplication was collapsing thousands
        # of BRAF measurements down to ~77 unique molecules - far too few to
        # train a QSAR model.
        print("\nDownloading activities (all records with a pChEMBL value)...\n")

        activities = self.activity_api.filter(
            target_chembl_id=target_chembl,
            standard_type=config.ACTIVITY_TYPE,
            pchembl_value__isnull=False,
        )

        all_rows = []
        offset = 0
        page_size = 1000
        consecutive_failures = 0

        while True:

            try:
                page = list(activities[offset:offset + page_size])
                consecutive_failures = 0
            except Exception as error:
                consecutive_failures += 1
                if consecutive_failures > 5:
                    raise Exception(
                        "ChEMBL download failed repeatedly; the EBI service may "
                        "be down. Try again later."
                    )
                print(f"Connection issue ({error}); retrying in 5s...")
                time.sleep(5)
                continue

            if not page:
                break

            all_rows.extend(page)
            print(f"Total downloaded: {len(all_rows)}")

            offset += page_size
            time.sleep(0.2)

        df = pd.DataFrame(all_rows)
        df.to_csv(config.RAW_DATASET, index=False)

        print("\nDownload complete!")
        print(f"Saved {len(df)} activity records")
        print(f"Location: {config.RAW_DATASET}")

    def run(self):

        target_chembl = self.resolve_target().chembl_id

        self.download_pages(target_chembl)

if __name__ == "__main__":

    downloader = ChEMBLDownloader()

    downloader.run()