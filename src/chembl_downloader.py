import time
import pandas as pd

from chembl_webresource_client.new_client import new_client

from config import TARGET, ACTIVITY_TYPE, MIN_COMPOUNDS, RAW_DATASET


class ChEMBLDownloader:

    def __init__(self):

        self.target_api = new_client.target
        self.activity_api = new_client.activity

    def search_target(self):

        print(f"\nSearching ChEMBL for '{TARGET}'...\n")

        targets = list(self.target_api.search(TARGET))

        if len(targets) == 0:
            raise Exception("No targets found.")

        print("Available targets:\n")

        for i, t in enumerate(targets):

            print(
                f"{i}: "
                f"{t['target_chembl_id']} | "
                f"{t.get('pref_name','Unknown')} | "
                f"{t.get('target_type','Unknown')}"
            )

        choice = int(input("\nChoose target number: "))

        target = targets[choice]

        print("\nSelected Target")

        print("-------------------------")

        print("ChEMBL ID :", target["target_chembl_id"])
        print("Name      :", target.get("pref_name"))
        print("Type      :", target.get("target_type"))

        return target["target_chembl_id"]

    def download_pages(self, target_chembl):

        print("\nDownloading activities...\n")

        all_rows = []

        offset = 0
        page_size = 500

        while len(all_rows) < MIN_COMPOUNDS:

            print(f"Downloading records {offset} - {offset + page_size}")

            try:

                activities = self.activity_api.filter(
                    target_chembl_id=target_chembl,
                    standard_type=ACTIVITY_TYPE
                )[offset:offset + page_size]

                page = list(activities)

                if len(page) == 0:
                    break

                all_rows.extend(page)

                print(f"Total downloaded: {len(all_rows)}")

                offset += page_size

                time.sleep(0.5)

            except Exception as e:

                print("Connection failed.")
                print(e)

                print("Retrying in 5 seconds...")

                time.sleep(5)

        df = pd.DataFrame(all_rows)

        if len(df) > MIN_COMPOUNDS:
            df = df.iloc[:MIN_COMPOUNDS]

        df.to_csv(RAW_DATASET, index=False)

        print("\nDownload complete!")
        print(f"Saved {len(df)} records")
        print(f"Location: {RAW_DATASET}")

    def run(self):

        target_chembl = self.search_target()

        self.download_pages(target_chembl)

if __name__ == "__main__":

    downloader = ChEMBLDownloader()

    downloader.run()