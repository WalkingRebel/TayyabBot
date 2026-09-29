import streamlit as st

from rag_pipeline import TayyabRAG


st.set_page_config(
    page_title="TayyabBot",
    page_icon="🤖",
    layout="centered"
)


st.title("🤖 TayyabBot")

st.write(
    "A personal RAG-based chatbot that "
    "answers questions using Tayyab's "
    "personal dataset."
)


@st.cache_resource
def load_rag():
    return TayyabRAG()


try:
    rag = load_rag()

except Exception as error:
    st.error(
        "TayyabBot could not start."
    )

    st.code(
        str(error)
    )

    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )


user_question = st.chat_input(
    "Ask TayyabBot something about Tayyab..."
)


if user_question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(
            user_question
        )

    with st.chat_message(
        "assistant"
    ):
        with st.spinner(
            "Searching Tayyab's personal dataset..."
        ):
            try:
                answer, sources = rag.ask(
                    user_question
                )

                st.markdown(answer)

                unique_sources = list(
                    dict.fromkeys(sources)
                )

                if unique_sources:
                    st.markdown(
                        "**Retrieved sources:**"
                    )

                    for source in unique_sources:
                        st.caption(
                            f"📄 {source}"
                        )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as error:
                st.error(
                    f"Error: {error}"
                )


with st.sidebar:

    st.header("About TayyabBot")

    st.write(
        """
        TayyabBot is a personalized
        Retrieval-Augmented Generation
        chatbot.

        It retrieves relevant information
        from a personal dataset and uses
        an LLM to generate contextual
        responses.
        """
    )

    st.divider()

    st.subheader("RAG Components")

    st.write(
        "✅ Personal Dataset"
    )

    st.write(
        "✅ Text Preprocessing"
    )

    st.write(
        "✅ Embeddings"
    )

    st.write(
        "✅ FAISS Vector Database"
    )

    st.write(
        "✅ Retrieval"
    )

    st.write(
        "✅ LLM Generation"
    )

    st.write(
        "✅ Prompt Engineering"
    )

    st.write(
        "✅ Conversation History"
    )