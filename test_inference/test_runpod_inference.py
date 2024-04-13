import os
import time
from pprint import PrettyPrinter

import runpod
from dotenv import load_dotenv

pprint = PrettyPrinter(indent=2)

load_dotenv()

runpod.api_key = os.getenv("RUNPOD_API_KEY")

endpoint = runpod.Endpoint(os.getenv("AUDIO_CONTENT_BUCKET_NAME"))


def synchronous_run():
    start_time = time.time()
    try:
        run_request = endpoint.run_sync(
            {
                "input": {
                    "target_lang": "lug",
                    "adapter": "lug",
                    "audio_file": "./content/SIMBA 10.1.mp3",
                }
            },
            timeout=600,  # Timeout in seconds.
        )

        pprint.pprint(run_request)
    except TimeoutError:
        print("Job timed out.")

    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")


if __name__ == "__main__":
    synchronous_run()
