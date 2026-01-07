import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Option 1: Free Gemini API (commented out - token limits too strict for our use)
# api_key = os.environ.get("GEMINI_API_KEY")
# print(f"Using key {api_key[:6]}...")
# client = genai.Client(api_key=api_key)

# Option 2: Vertex AI (currently using this approach)
project = os.environ.get("GEMINI_PROJECT")
location = os.environ.get("GEMINI_LOCATION")
print(f"Using project {project} and location {location}...")

client = genai.Client(vertexai=True, project=project, location=location)
model_name = "gemini-2.0-flash-001"
contents = (
    "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
)


def main():
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
    )

    print(response.text)
    print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
    print(f"Response Tokens: {response.usage_metadata.candidates_token_count}")


if __name__ == "__main__":
    main()
