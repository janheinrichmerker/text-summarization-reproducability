module Datasets

using DataDeps

include("./download_utils.jl")

function __init__()
    register(
        DataDep(
            "XSum-Preprocessed-BERT",
            """
            Preprocessed XSum summary data for BERT-based summarization.
            """,
            "1BWBN1coTWGBqrWoOfRc5dhojPHhatbYs",
            "26bc6ff7dafe040fc3a14578e4dbb9193f276d34f1e7a310edb44b8bfd77384c",
            fetch_method=fetch_google_drive_id,
            post_fetch_method=unpack,
        )
    )

    register(
        DataDep(
            "CNN-Dailymail-Preprocessed-BERT",
            """
            Preprocessed CNN/Dailymail summary data for BERT-based summarization.
            """,
            "1DN7ClZCCXsk2KegmC6t4ClBwtAf5galI",
            fetch_method=fetch_google_drive_id,
            post_fetch_method=unpack,
        )
    )
end

# TODO Preprocess data for Transformers.jl.

end
