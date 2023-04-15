package com.video.tagger;

import org.springframework.stereotype.Service;


@Service
public class ContentServiceImpl implements IcontentService{
	

	@Override
	public String getTags(String url) {
		// TODO Auto-generated method stub 
		ShellCommandRunner runner = new ShellCommandRunner();
		
		return runner.runShell(url);
		//return "tour de france, competition, cycli ng";
	}
	@Override
	public String getTranscribedText(String url) {
		// TODO Auto-generated method stub 
		ShellCommandRunner runner = new ShellCommandRunner();
		
		return runner.runGetTranscribedText(url);
	}
	
	@Override
	public void transcribeVideo(String url)  {
		// TODO Auto-generated method stub 
		ShellCommandRunner runner = new ShellCommandRunner();
		
		runner.runTranscribeVideo(url);

	}
	@Override
	public String getTranscribedTags(String url) {
		ShellCommandRunner runner = new ShellCommandRunner();
		return runner.runGetTranscribedCsv(url);

	}
	
	
	

}
