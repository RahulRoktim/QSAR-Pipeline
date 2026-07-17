import time
import pandas as pd

from chembl_webresource_client.new_client import new_client

from config import (
    TARGET,
    ACTIVITY_TYPE,
    RAW_DATASET,
    TARGET_CHEMBL_ID,
)


class ChEMBLDownloader:

    def __init__(self):

        self.target_api = new_client.target
        self.activity_api = new_client.activity

    def search_target(self):

        # Automatable: use a configured ChEMBL id if provided, otherwise search
        # and auto-pick a single-protein target (no interactive input()).
        if TARGET_CHEMBL_ID:
            print(f"Using configured target: {TARGET_CHEMBL_ID}")
            return TARGET_CHEMBL_ID

        print(f"\nSearching ChEMBL for '{TARGET}'...\n")

        targets = None
        for attempt in range(1, 6):
            try:
                targets = list(self.target_api.search(TARGET))
                break
            except Exception as error:
                print(
                    f"ChEMBL target search failed (attempt {attempt}/5): {error}"
                )
                print(
                    "The ChEMBL/EBI service may be temporarily unavailable. "
                    "Retrying in 10s..."
                )
                time.sleep(10)

        if not targets:
            raise Exception(
                "Could not reach ChEMBL after several attempts. The EBI service "
                "may be down (e.g. HTTP 500) - try again later, or set "
                "TARGET_CHEMBL_ID in config.py to skip the target search."
            )

        for i, t in enumerate(targets):
            print(
                f"{i}: {t['target_chembl_id']} | "
                f"{t.get('pref_name','Unknown')} | "
                f"{t.get('target_type','Unknown')}"
            )

        # ChEMBL often has several target entries for one protein, and the
        # minor duplicates carry almost no data (this is exactly how the wrong
        # BRAF entry with only ~77 activities got picked). Among single-protein
        # targets, choose the one with the MOST activity records for our
        # endpoint.
        single_protein = [
            t for t in targets if t.get("target_type") == "SINGLE PROTEIN"
        ]
        candidates = single_protein or targets

        def activity_count(target_id):
            try:
                return len(
                    self.activity_api.filter(
                        target_chembl_id=target_id,
                        standard_type=ACTIVITY_TYPE,
                        pchembl_value__isnull=False,
                    )
                )
            except Exception:
                return 0

        chosen = max(
            candidates,
            key=lambda t: activity_count(t["target_chembl_id"]),
        )

        print(
            f"\nAuto-selected: {chosen['target_chembl_id']} | "
            f"{chosen.get('pref_name')} | {chosen.get('target_type')} "
            "(most activities)"
        )

        return chosen["target_chembl_id"]

    def download_pages(self, target_chembl):

        # Download ALL activities that carry a pChEMBL value (not a capped 1000).
        # The previous cap + downstream de-duplication was collapsing thousands
        # of BRAF measurements down to ~77 unique molecules - far too few to
        # train a QSAR model.
        print("\nDownloading activities (all records with a pChEMBL value)...\n")

        activities = self.activity_api.filter(
            target_chembl_id=target_chembl,
            standard_type=ACTIVITY_TYPE,
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
        df.to_csv(RAW_DATASET, index=False)

        print("\nDownload complete!")
        print(f"Saved {len(df)} activity records")
        print(f"Location: {RAW_DATASET}")

    def run(self):

        target_chembl = self.search_target()

        self.download_pages(target_chembl)

if __name__ == "__main__":

    downloader = ChEMBLDownloader()

    downloader.run()