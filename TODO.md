- Top TODO (for now): Go ahead and work on deployment, even though there is plenty of retrieval work to do.
Getting an actual app to play with early will help with core logic decision making.



- Create multi-granularity approach: we want both big chunks for more general queries and smaller chunks for more specific ones.
- Make it so that we can encode the answer according to our training data and compare similarity between ground truth answer and the query under test.
    - Critical for determining if the problem is our chunking system or embedding quality.
- Start building out the UI.
    - I'm leaning towards a simple, static web app, with a simple, clean design.
- Create a test results analyzer which takes in test results and calculates some stats on how well the retrieval went.
- Make the final results a weighted sum of keyword match and semantic similarity match.
    - Give the user a slider that allows them to adjust weights in realtime. New results should pop up on their screen as they adjust the slider.
- Consider training an ML model to assign weights dynamically between semantic and keyword component.