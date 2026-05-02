from llm4rec.data.readiness import NO_EXECUTION_FLAG
from llm4rec.experiments.job_queue import create_job_queue
from llm4rec.experiments.launch_manifest import create_launch_manifest


def test_job_queue_writes_planned_jobs_only(tmp_path):
    manifest = create_launch_manifest(tmp_path / "launch_manifest.json")
    jobs = create_job_queue(manifest, tmp_path)
    assert jobs
    assert all(job["status"] == "planned" for job in jobs)
    assert all(job["allow_api_calls"] is False for job in jobs)
    assert all(job[NO_EXECUTION_FLAG] is True for job in jobs)
    assert (tmp_path / "jobs.jsonl").is_file()
    assert (tmp_path / "jobs.csv").is_file()
    assert (tmp_path / "go_no_go_checklist.md").is_file()
