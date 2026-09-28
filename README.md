# CSC-128 Module 6 - Grounded Policy Bot

## Project Overview

For this assignment I created a grounded policy bot that answers questions using information from my knowledge base. The bot uses TF-IDF and cosine similarity to find relevant chunks before sending anything to the language model.

If the retriever does not find relevant information, the bot returns the refusal message instead of calling the model. This helps prevent the bot from making up answers.

## Retrieval Testing

I tested the retriever with questions that should match information in my knowledge base. I also tested questions that were not covered by the knowledge base.

I adjusted the retrieval threshold after looking at the similarity scores. At first, some unrelated questions were getting weak matches. I changed the threshold so those questions would return no relevant information instead.

I also wrote the knowledge chunks using wording that students would actually use when searching for library information.

## Hallucination Tests

I tested five questions that are not answered by my knowledge base.

### Test 1
Question: Can I bring my dog into the library?

Actual response: I don't have that information in my knowledge base.

Result: The bot refused because there was no relevant information retrieved.

### Test 2
Question: How much does tuition cost?

Actual response: I don't have that information in my knowledge base.

Result: The bot refused because tuition information is not in the knowledge base.

### Test 3
Question: What food is served in the cafeteria?

Actual response: I don't have that information in my knowledge base.

Result: The bot refused because cafeteria information is not included.

### Test 4
Question: Where do I park my car?

Actual response: I don't have that information in my knowledge base.

Result: The bot refused because parking information is not included.

### Test 5
Question: Can my professor change my final grade?

Actual response: I don't have that information in my knowledge base.

Result: The bot refused because grading policies are not included.

## What I Changed

During testing, I noticed that a lower retrieval threshold could allow unrelated chunks through. I adjusted the threshold based on the similarity scores from my tests instead of just guessing.

I also made the chunks use student-friendly wording so questions are more likely to match the correct information.

The grounding prompt tells the model to only use the retrieved context and not outside knowledge. The short circuit happens before the model is called, so if retrieval returns nothing the bot refuses immediately.

## Source Attribution

When the bot answers a supported question, it displays the source for the retrieved information. This helps me tell the difference between a retrieval problem and a generation problem. If the wrong source is retrieved, the problem is probably with the threshold or chunk wording. If the correct source is retrieved but the answer is wrong, the problem is more likely with the grounding prompt.