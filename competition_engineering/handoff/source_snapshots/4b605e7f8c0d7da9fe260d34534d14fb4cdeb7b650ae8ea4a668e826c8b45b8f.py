"""Bounded retries for a pre-worker transient namespace allocation failure."""
import time

NAMESPACE_TRANSIENT=b'bwrap: Creating new namespace failed: Resource temporarily unavailable'

def training_with_namespace_retry(launch,deadline,on_retry,max_attempts=3,sleep=time.sleep):
    for attempt in range(1,max_attempts+1):
        process,status,watcher=launch()
        stdout,stderr=process.communicate(timeout=max(1,deadline-time.monotonic()+10));watcher.join(timeout=1)
        transient=process.returncode==1 and not status and not stdout and stderr.strip()==NAMESPACE_TRANSIENT
        if not transient or attempt==max_attempts or deadline-time.monotonic()<=1:
            return process,status,stdout,stderr
        on_retry(attempt,stderr);sleep(1)
    raise AssertionError('Unreachable retry state')
