import argparse
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_response(prompt, args):
    response = openai.Completion.create(
        model = args.model,
        prompt = prompt,
        temperature = args.temperature,
        top_p = args.top_p,
        max_tokens = 20,
    )

    return response

def store_responses():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='choose model to prompt', choices=['gpt-3.5-turbo', 'galactica-3B'])
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default = 1.0)

    args = parser.parse_args()

    # read prompts

    # get responses

    # store responses