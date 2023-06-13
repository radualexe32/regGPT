import argparse
import os

def main(api_key):
    with open('.env', 'w') as f:
        f.write(f'OPENAI_API_KEY={api_key}\n')
    print(".env file successfully generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate .env file with OpenAI API key')
    parser.add_argument('api_key', type = str, help = 'The OpenAI API key')
    args = parser.parse_args()
    main(args.api_key)