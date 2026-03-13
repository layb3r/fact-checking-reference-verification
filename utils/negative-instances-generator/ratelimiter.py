from datetime import datetime, timedelta
import logging, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

class RateLimiter:
    """Track and enforce rate limits for Gemma 27B API calls."""
    
    def __init__(self, requests_per_minute: int = 30, tokens_per_minute: int = 15000):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        
        # Tracking windows (1 minute rolling)
        self.request_times = []
        self.token_usage = []  # (timestamp, token_count)
        
        # Statistics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
    def wait_if_needed(self):
        """Sleep if we're approaching rate limits."""
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        # Clean old entries
        self.request_times = [t for t in self.request_times if t > one_minute_ago]
        self.token_usage = [(t, c) for t, c in self.token_usage if t > one_minute_ago]
        
        # Check request rate
        recent_requests = len(self.request_times)
        if recent_requests >= self.requests_per_minute - 2:  # Leave 2 request buffer
            sleep_time = 62  # Wait slightly over a minute
            log.warning(f"Approaching request limit ({recent_requests}/{self.requests_per_minute}). Sleeping {sleep_time}s...")
            time.sleep(sleep_time)
            return
        
        # Check token rate
        recent_tokens = sum(c for _, c in self.token_usage)
        if recent_tokens >= self.tokens_per_minute - 1000:  # Leave 1K token buffer
            sleep_time = 62
            log.warning(f"Approaching token limit ({recent_tokens}/{self.tokens_per_minute}). Sleeping {sleep_time}s...")
            time.sleep(sleep_time)
            return
    
    def record_request(self, input_tokens: int = 0, output_tokens: int = 0):
        """Record a successful API call."""
        now = datetime.now()
        self.request_times.append(now)
        
        total_tokens = input_tokens + output_tokens
        self.token_usage.append((now, total_tokens))
        
        self.total_requests += 1
        self.total_tokens += total_tokens
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
    
    def log_stats(self):
        """Log cumulative usage statistics."""
        log.info("=" * 60)
        log.info("API Usage Statistics:")
        log.info(f"  Total Requests:      {self.total_requests}")
        log.info(f"  Total Tokens:        {self.total_tokens:,}")
        log.info(f"    Input Tokens:      {self.total_input_tokens:,}")
        log.info(f"    Output Tokens:     {self.total_output_tokens:,}")
        log.info(f"  Avg Tokens/Request:  {self.total_tokens / max(1, self.total_requests):.1f}")
        log.info("=" * 60)
