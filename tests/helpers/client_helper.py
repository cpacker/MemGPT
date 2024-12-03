import time
from typing import Union

from letta import LocalClient, RESTClient
from letta.schemas.enums import JobStatus
from letta.schemas.job import Job
from letta.schemas.source import Source


def upload_file_using_client(client: Union[LocalClient, RESTClient], source: Source, filename: str) -> Job:
    # load a file into a source (non-blocking job)
    upload_job = client.load_file_to_source(filename=filename, source_id=source.id, blocking=False)
    print("Upload job", upload_job, upload_job.status, upload_job.metadata_)

    # view active jobs
    active_jobs = client.list_active_jobs()
    jobs = client.list_jobs()
    assert upload_job.id in [j.id for j in jobs]
    assert len(active_jobs) == 1
    assert active_jobs[0].metadata_["source_id"] == source.id

    # wait for job to finish (with timeout)
    timeout = 240
    start_time = time.time()
    while True:
        status = client.get_job(upload_job.id).status
        print(f"\r{status}", end="", flush=True)
        if status == JobStatus.completed:
            break
        time.sleep(1)
        if time.time() - start_time > timeout:
            raise ValueError("Job did not finish in time")

    return upload_job
