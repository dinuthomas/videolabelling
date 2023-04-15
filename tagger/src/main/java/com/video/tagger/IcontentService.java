package com.video.tagger;

public interface IcontentService {
	
	String getTags(String url) ;
	void transcribeVideo(String url) ;
	String getTranscribedText(String url) ;
	String getTranscribedTags(String url) ;

}
