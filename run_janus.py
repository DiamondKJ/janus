import argparse, logging, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from janus.mind import JanusMind
def main():
    parser = argparse.ArgumentParser(description="Awaken the Janus-Core Cognitive Architecture.")
    parser.add_argument("--api_key", type=str, required=True, help="Your Google AI API key.")
    args = parser.parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    janus = JanusMind(project_root, args.api_key)
    janus.start_interactive_loop()
if __name__ == "__main__":
    main()