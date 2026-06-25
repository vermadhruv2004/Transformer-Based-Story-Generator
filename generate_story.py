from transformers import pipeline

MODEL_PATH = "model/fine_tuned_model"

print("Loading model...")

generator = pipeline(
    "text-generation",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH
)

print("Model loaded successfully!")


def generate_story(prompt, temperature=0.8, top_p=0.9):

    result = generator(
    f"### Instruction:\n{prompt}\n\n### Story:\n",
    max_new_tokens=400,
    temperature=0.9,
    top_p=0.92,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    do_sample=True
)

    story = result[0]["generated_text"]

    if "### Story:" in story:
        story = story.split("### Story:")[-1].strip()

    return story


# Testing
if __name__ == "__main__":

    prompt = input("Enter Prompt: ")

    output = generate_story(prompt)

    print("\nGenerated Story:\n")
    print(output)