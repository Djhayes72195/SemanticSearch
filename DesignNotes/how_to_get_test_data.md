At this point in the development process, I believe that the test runner is robust enough to begin thinking about gathering good, representative test data. We want data that:

- Is similar to notes that a student or professional might take.
- Is varied in terms of organization. We should be able to handle bullet points, paragraphs, essays
and everything in between.
- Is varied in terms of content. A student might have notes on a diverse set of classes, and a professional could have notes on workplace systems, domain knowledge, online class notes, etc.
- Realistically large. We need to get an idea of how well this app will scale. I don't think we need to be prepared to process millions of files, but we should be able to comfortably handle a large enough set.


Not every test set has to meet each of these criteria, but our complete set of test sets should cover all cases.

In addition to having the test data itself, we need a set of questions and answers in order to systematically gauge how well the app is working. I can think of a few way to generate question and answer sets.

1. By hand method 1: I could go through the corpus of choice by hand, find passages that answer some question or contain key information, and create a query that I think should return that passage.
2. Programmatic method 1: I could split the corpus and feed a random sample of passaged to a paraphraser. I could instruct the paraphraser to rephrase the passage into a format that a user might input as a query.
3. By hand method 2: I could outsource the QA generation to others. Using either a corpus that I provide or a corpus of their own, ask others to generate question/answer pairs. If I can convince anyone else to do this that would be ideal so that I don't overfit this app to my own personal preferences.
4. Programmatic method 2: I could feed passages to a more sophisticated model (GPT, BERT) and instruct it to generate questions for a particular passage or document.
5. Use SQuAD dataset: The SQuAD dataset contains 100,000 question and answer pairs based on a set of wikipeedia articles. Half of the questions are unanswerable given the source data, which is could be very useful for tuning our system to determine when there is no suitable answer to the query posed.
6. Just trying the app out. I have work notes myself, and a reasonably good idea of what I might find in there (just as a real user would). I could feed it my own work notes and experience the app as a user would.

A blend of approaches should be used to ensure that I am not overfitting to any particular method. Some methods also might be more appropriate some datasets and not others.