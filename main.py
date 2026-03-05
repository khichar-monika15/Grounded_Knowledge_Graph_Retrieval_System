"""Root entrypoint — delegates to pipeline.run_pipeline."""
from pipeline.run_pipeline import run_pipeline

def main():
    run_pipeline()

if __name__ == "__main__":
    main()
