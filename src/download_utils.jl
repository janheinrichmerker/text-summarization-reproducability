# Adapted from Transformers.jl and HTTP.jl, both released under the MIT License.
# Original files:
# https://github.com/chengchingwen/Transformers.jl/blob/27a736794487f303d7919ac03623c9d34670f744/src/datasets/download_utils.jl
# https://github.com/JuliaWeb/HTTP.jl/blob/ed02d37b373386804707262d68856c0fad329b16/src/download.jl

using HTTP
using Dates
using DataDeps
using Random:randstring

const google_base_domain = "docs.google.com"


function fetch_google_drive_id(drive_id, local_path)
    url = "https://$google_base_domain/uc?id=$drive_id&export=download"
    fetch_google_drive(url, local_path)
end

function find_google_confirmation_code(cookiejar::Dict{String,Set{HTTP.Cookies.Cookie}})
    cookies = cookiejar[google_base_domain]
    cookies = filter(
        cookie -> startswith(cookie.name, "download_warning_"), 
        cookies
    )
    first(cookies).value
end

function confirm_google_download(url, cookiejar)
    HTTP.request("HEAD", url; cookies=true, cookiejar=cookiejar)
    code = find_google_confirmation_code(cookiejar)
    if isnothing(code)
        error("download failed")
    end
    "$url&confirm=$code"
end

function fetch_google_drive(url, localdir)
    cookiejar = Dict{String,Set{HTTP.Cookies.Cookie}}()
    url = confirm_google_download(url, cookiejar)

    format_progress(x) = round(x, digits=4)
    format_bytes(x) = !isfinite(x) ? "∞ B" : Base.format_bytes(x)
    format_seconds(x) = "$(round(x; digits=2)) s"
    format_bytes_per_second(x) = format_bytes(x) * "/s"


    filepaths::Array{String} = []

    HTTP.open(
        "GET",
        url,
        ["Range" => "bytes=0-"];
        cookies=true,
        cookiejar=cookiejar,
        redirect_limit=10
    ) do stream
        response = HTTP.startread(stream)
        content_disposition = HTTP.header(response, "Content-Disposition")

        filename_match = match(r"filename=\\\"(.*)\\\"", content_disposition)
        filename = filename_match !== nothing ? filename_match.captures[] : basename(tempname())
        filepath = joinpath(localdir, filename)

        parsed_total_bytes::Union{Number,Nothing} = tryparse(
            Float64, 
            split(
                HTTP.header(response, "Content-Range"), 
                '/'
            )[end]
        )
        total_bytes::Number = parsed_total_bytes !== nothing ? parsed_total_bytes : NaN
        downloaded_bytes::Number = 0
        start_time::DateTime = Dates.now()
        prev_time::DateTime = Dates.now()
        period::Number = DataDeps.progress_update_period()

        function report_callback()
            if filename_match === nothing && isnan(total_bytes)
                # Skip reporting intermediate redirects.
                return
            end

            prev_time = Dates.now()
            taken_time = (prev_time - start_time).value / 1000
            average_speed = downloaded_bytes / taken_time
            remaining_bytes = total_bytes - downloaded_bytes
            remaining_time = remaining_bytes / average_speed
            completion_progress = downloaded_bytes / total_bytes

            @info(
                "Downloading",
                source = url,
                dest = filepath,
                progress = completion_progress |> format_progress,
                time_taken = taken_time |> format_seconds,
                time_remaining = remaining_time |> format_seconds,
                average_speed = average_speed |> format_bytes_per_second,
                downloaded = downloaded_bytes |> format_bytes,
                remaining = remaining_bytes |> format_bytes,
                total = total_bytes |> format_bytes,
            )
        end

        Base.open(filepath, "w") do fh
            while (!eof(stream))
                downloaded_bytes += write(fh, readavailable(stream))
                if !isinf(period)
                    if Dates.now() - prev_time > Millisecond(1000 * period)
                        report_callback()
                    end
                end
            end
        end
        if !isinf(period)
            report_callback()
        end

        push!(filepaths, filepath)
    end

    final_filepath = pop!(filepaths)
    for path ∈ filepaths
        # Remove intermediate redirects.
        rm(path)
    end
    final_filepath
end
