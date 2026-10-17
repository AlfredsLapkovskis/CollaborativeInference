import asyncio
import websockets
import time


class RateLimitedClientConnection(websockets.ClientConnection):
    @staticmethod
    def create_factory(send_rate_limit=None, recv_rate_limit=None):
        def create_connection(*args, **kwargs):
            return RateLimitedClientConnection(*args, **kwargs, send_rate_limit=send_rate_limit, recv_rate_limit=recv_rate_limit)
        return create_connection

    def __init__(self, protocol, *, ping_interval=20, ping_timeout=20, close_timeout=10, max_queue=16, write_limit=2 ** 15, send_rate_limit=None, recv_rate_limit=None):
        super().__init__(protocol, ping_interval=ping_interval, ping_timeout=ping_timeout,
                         close_timeout=close_timeout, max_queue=max_queue, write_limit=write_limit)

        self.send_rate_limit = send_rate_limit
        self.recv_rate_limit = recv_rate_limit

        self.last_send_time = time.monotonic()
        self.last_recv_time = time.monotonic()

        self.send_tokens = send_rate_limit
        self.recv_tokens = recv_rate_limit

    async def send(self, message, text=None):
        data_len = len(message)
        current_time = time.monotonic()

        if self.send_rate_limit:
            elapsed = current_time - self.last_send_time

            self.send_tokens += elapsed * self.send_rate_limit
            if self.send_tokens > self.send_rate_limit:
                self.send_tokens = self.send_rate_limit

            if data_len > self.send_tokens:
                deficit = data_len - self.send_tokens
                wait_time = deficit / self.send_rate_limit
                try:
                    await asyncio.sleep(wait_time)
                except BaseException as exc:
                    print(f"Send with rate limit {exc=}")
                    raise exc
                finally:
                    self.send_tokens = 0
                    self.last_send_time = current_time + wait_time
            else:
                self.send_tokens -= data_len
                self.last_send_time = current_time
        else:
            self.last_send_time = current_time

        return await super().send(message, text)

    async def recv(self, decode: bool | None = None):
        try:
            message = await super().recv(decode)
        except BaseException as exc:
            raise exc
        finally:
            data_len = len(message)
            current_time = time.monotonic()

            if self.recv_rate_limit:
                elapsed = current_time - self.last_recv_time

                self.recv_tokens += elapsed * self.recv_rate_limit
                if self.recv_tokens > self.recv_rate_limit:
                    self.recv_tokens = self.recv_rate_limit

                if data_len > self.recv_tokens:
                    deficit = data_len - self.recv_tokens
                    wait_time = deficit / self.recv_rate_limit
                    try:
                        await asyncio.sleep(wait_time)
                    except BaseException as exc:
                        print(f"Recv with rate limit {exc=}")
                        raise exc
                    finally:
                        self.recv_tokens = 0
                        self.last_recv_time = current_time + wait_time
                else:
                    self.recv_tokens -= data_len
                    self.last_recv_time = current_time
            else:
                self.last_recv_time = current_time

        return message
