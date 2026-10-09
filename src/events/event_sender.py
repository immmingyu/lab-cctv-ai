import queue
import threading
import time
import requests

class EventSender:
    """
    이벤트를 Backend API로 보내는 역할.
    별도 스레드에서 전송하므로, 서버가 느리거나 꺼져 있어도 영상 처리가 멈추지 않음.
    """

    def __init__(self, url="http://127.0.0.1:8000/events", timeout=2):
        self.url = url
        self.timeout = timeout
        self.q = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def send(self, event):
        self.q.put(event)  # 큐에 넣기만 하고 바로 반환

    def _run(self):
        while True:
            event = self.q.get()
            try:
                r = requests.post(self.url, json=event, timeout=self.timeout)
                if r.status_code != 200:
                    print(f"[Sender] 서버가 거절함 ({r.status_code}): {r.text}")
            except requests.RequestException as e:
                print(f"[Sender] 전송 실패 (서버가 꺼져 있나요?): {e}")
            finally:
                self.q.task_done()

    def flush(self, timeout=3):
        """프로그램 종료 직전, 남은 이벤트가 전송될 시간을 잠깐 줌."""
        start = time.time()
        while self.q.unfinished_tasks > 0 and time.time() - start < timeout:
            time.sleep(0.1)