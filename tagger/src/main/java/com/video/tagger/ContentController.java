package com.video.tagger;

import java.util.concurrent.TimeUnit;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
//import org.springframework.web.bind.annotation.RestController;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestParam;



@Controller
@RequestMapping("/content")
public class ContentController {

	@Autowired
	private ContentServiceImpl contentService;
	
	private VideoToText transcribe = null;
	
	private String videoUrl = null;
	

	@GetMapping("/videotag")
	public String getVideoTagInfo(@RequestParam(name="url", required=false, defaultValue="") String url, Model model) {
		System.out.println(url);
		String output = contentService.getTags(url);
		System.out.println("**************************");
		System.out.println(output);
		model.addAttribute("output", output);
		return "videotag";
	}
	
	@GetMapping("/greeting")
	public String greeting(@RequestParam(name="name", required=false, defaultValue="World") String name, Model model) {
		model.addAttribute("name", name);
		return "greeting";
	}
	  @GetMapping("/video2text")
	  public String greetingForm(@ModelAttribute VideoToText video2text, Model model) {
		  System.out.println("ContnetControl" + "control comes in get mapping"); 
		  this.transcribe = new VideoToText();
		  model.addAttribute("video2text", this.transcribe);
		  

	    return "video2text";
	  }
	
	  @PostMapping("/video2text")
	  public String greetingSubmit(@ModelAttribute VideoToText video2text, Model model) {
		System.out.println("ContnetControl" + "control comes in post mapping"); 

		model.addAttribute("video2text", video2text);
		
		String videoUrl = video2text.getId();
		contentService.transcribeVideo(videoUrl);
		this.videoUrl = videoUrl;
		
		String asrText = contentService.getTranscribedText(videoUrl);
		System.out.println(asrText);
		video2text.setContent(asrText);
		String keywords = contentService.getTranscribedTags(videoUrl);
		System.out.println(keywords);
		video2text.setKeywords(keywords);

	    return "results";
	  }
	  
	  
	//http://localhost:8080/content/videotag

}

/**
 * web : url
 * */
