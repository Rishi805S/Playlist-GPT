# Before used Youtube Transcript Loader which was a Document loader but that wasn't giving timestamps it was only providing content so I had to switch to this new Youtube Transcript APi


from youtube_transcript_api import YouTubeTranscriptApi

ytt_api = YouTubeTranscriptApi()
print(ytt_api.fetch("nhLZKsxEwxM")[:2])