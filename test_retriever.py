# CSC-128 Module 6
# Tests for my retriever

from retriever import Retriever, analyze


retriever = Retriever()


def show_results(question):
    print("\n" + "=" * 60)
    print("QUESTION:", question)

    hits = retriever.search(question)

    if not hits:
        print("RESULT: No relevant information found.")
        return

    print("RESULTS:")

    for document, score in hits:
        print(
            f"- {document['id']} | "
            f"score: {score:.3f} | "
            f"source: {document['source']}"
        )


# First I checked the analyzer to make sure it was
# normalizing words and creating bigrams.
print("ANALYZER TEST:")
print(analyze("Can I use computers at the library?"))


# Questions that should find something useful
print("\nRELEVANT QUESTION TESTS")

show_results("Can I use a computer at the library?")
show_results("How do I find articles for my assignment?")
show_results("Can I use library databases from home?")
show_results("Where can I study?")
show_results("Can I check out books?")


# Near-miss questions are important because the bot should
# not answer questions that are outside of its knowledge base.
print("\nNEAR-MISS / HALLUCINATION TESTS")

show_results("Can I bring my dog into the library?")
show_results("How much does tuition cost?")
show_results("What food is served in the cafeteria?")
show_results("Where do I park my car?")
show_results("Can my professor change my final grade?")


# Show what context would be sent to the model
print("\n" + "=" * 60)
print("CONTEXT TEST")

context_hits = retriever.search(
    "Can I get research help for an assignment?"
)

print(retriever.build_context(context_hits))