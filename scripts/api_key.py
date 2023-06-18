import argparse
import os


def read_existing_keys(file_path):
    existing_keys = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    key, value = line.split('=')
                    existing_keys[key] = value
    return existing_keys


def write_to_env(file_path, existing_keys, **kwargs):
    for key, value in kwargs.items():
        if value is not None:
            existing_keys[key] = value

    with open(file_path, 'w') as f:
        for key, value in existing_keys.items():
            f.write(f'{key}={value}\n')
    print(".env file successfully updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate .env file with API keys')

    parser.add_argument('--openai', type=str, required=False,
                        help='The OpenAI API key')

    args = parser.parse_args()

    existing_keys = read_existing_keys('.env')

    write_to_env('.env',
                 existing_keys,
                 OPENAI_API_KEY=args.openai,
                 )
