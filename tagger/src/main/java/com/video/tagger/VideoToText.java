package com.video.tagger;

import org.springframework.beans.factory.annotation.Autowired;

public class VideoToText {

	  private String id;
	  private String content;
	  private String keywords;

	  public String getId() {
		System.out.println("VideoToText" + "control comes getId"); 
	    return id;
	  }

	  public void setId(String id) {
		System.out.println("VideoToText" + "control comes setId"); 
	    this.id = id;
	  }

	  public String getContent() {
	    return content;
	  }

	  public void setContent(String content) {
	    this.content = content;
	  }
	  
	  public String getKeywords() {
		    return keywords;
		  }

	  public void setKeywords(String keywords) {
		    this.keywords = keywords;
		  }
	  
}
