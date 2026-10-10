from uav_llm_demo import create_app
from uav_llm_demo.provider_usage import configure_usage_logging
from uav_llm_demo.run_log import configure_run_logging


configure_usage_logging()
configure_run_logging()
app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
