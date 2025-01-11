Annoy stores in embeddings by ID.

I have to map that ID back to the document from which is was embedded.

Eventually, I will also have to map that ID back to the specific section of the document
which corresponds to that embedding.

Right now I am splitting by sentence and new line, but I will be introducing other
ways and want the flexibility to introduce pretty much any splitting method
I can imagine.

I could start by storing all mappings - Document and section of document.
For preliminary testing I could only use the document itself.

What is the best way to do that?

ideas:
    - I could record the splitting method used to produce the embedding,
    record the # of the split under the strategy. When I want to find the text
    again I apply the same method and just use the # of split that is the highest
    match.
        - I think this is not a good approach. It seems too expensive and
        unneccesary.
    - I could count characters. The id mapping might looks like.
        {1: location_object}
        location_object contains char #s, source document name, source document location
        in user's file system.