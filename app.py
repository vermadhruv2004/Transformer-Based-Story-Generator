import gradio as gr

from generate_story import generate_story


def create_story(prompt,
                 temperature,
                 top_p):

    return generate_story(
        prompt,
        temperature,
        top_p
    )


with gr.Blocks(
    title="AI Story Generator"
) as demo:

    gr.Markdown(
        "# 📖 AI Story Generator"
    )

    gr.Markdown(
        "Generate creative stories using a fine-tuned Transformer model."
    )

    prompt = gr.Textbox(
        label="Story Prompt",
        placeholder="Write a fantasy story about a dragon kingdom..."
    )

    temperature = gr.Slider(
        minimum=0.1,
        maximum=1.5,
        value=0.9,
        step=0.1,
        label="Temperature"
    )

    top_p = gr.Slider(
        minimum=0.1,
        maximum=1.0,
        value=0.92,
        step=0.01,
        label="Top-P"
    )

    generate_btn = gr.Button(
        "Generate Story"
    )

    output = gr.Markdown(
    label="Generated Story"
    )

    generate_btn.click(
        fn=create_story,
        inputs=[
            prompt,
            temperature,
            top_p
        ],
        outputs=output
    )

demo.launch()