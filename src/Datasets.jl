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
            "e1921b399012b4ce7e585eabe8a73fea1e9541fc30892b21de2c503ab9a1f9a9",
            fetch_method=fetch_google_drive_id,
            post_fetch_method=unpack,
        )
    )
end

# TODO Preprocess data for Transformers.jl.

end
