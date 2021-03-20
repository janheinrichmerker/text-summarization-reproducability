using DataDeps

include("utils/download_utils.jl")

register(
    DataDep(
        "XSum-Preprocessed-BERT",
        """
        Preprocessed XSum summary data for BERT-based summarization.

        XSum dataset, preprocessed for summarization by Liu et al.
        Original and preprocessed dataset released by the University of Edinburgh.
            
        Citing:
        @inproceedings{narayan2018xsum,
            author={Shashi Narayan and Shay B. Cohen and Mirella Lapata},
            title={Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization},
            booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
            year={2018},
            address={Brussels, Belgium},
        }
        @inproceedings{liu2019presumm,
            author={Yang Liu and Mirella Lapata},
            title={Text Summarization with Pretrained Encoders},
            booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing}
            year={2019},
            address={Hong Kong, China},
        }
        """,
        "1BWBN1coTWGBqrWoOfRc5dhojPHhatbYs",
        "26bc6ff7dafe040fc3a14578e4dbb9193f276d34f1e7a310edb44b8bfd77384c",
        fetch_method=fetch_google_drive_id,
        post_fetch_method=unpack,
    )
)

register(
    DataDep(
        "CNN-Daily-Mail-Preprocessed-BERT",
        """
        Preprocessed CNN / Daily Mail summary data for BERT-based summarization.

        CNN / Daily Mail dataset, preprocessed for summarization by Liu et al.
        Original dataset released by Google DeepMind and the University of Oxford.
            
        Citing:
        @inproceedings{hermann2015cnndm,
            author={Karl Moritz Hermann and Tom\\'a\\v{s} Ko\\v{c}isk\\'y and Edward Grefenstette and Lasse Espeholt and Will Kay and Mustafa Suleyman and Phil Blunsom},
            title={Teaching Machines to Read and Comprehend},
            booktitle={Advances in Neural Information Processing Systems},
            year={2015},
            address={Montreal, Quebec, Canada},
        }
        @inproceedings{liu2019presumm,
            author={Yang Liu and Mirella Lapata},
            title={Text Summarization with Pretrained Encoders},
            booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing}
            year={2019},
            address={Hong Kong, China},
        }
        """,
        "1DN7ClZCCXsk2KegmC6t4ClBwtAf5galI",
        "e1921b399012b4ce7e585eabe8a73fea1e9541fc30892b21de2c503ab9a1f9a9",
        fetch_method=fetch_google_drive_id,
        post_fetch_method=unpack,
    )
)
