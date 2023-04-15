package com.video.tagger;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

public class ShellCommandRunner {
	
	    public  String runShell(String url)
	    {
	        try {
	  
	            System.out.println(
	                System.getProperty("os.name"));
	            System.out.println();
	            
	            System.out.println("Command = "+"/opt/iproject/videolabelling/asrlabels/transcribemany.sh"+" "+url);
	           try {
					Runtime.getRuntime().exec(new String[]{"/opt/iproject/videolabelling/asrlabels/transcribemany.sh","-c", url}).waitFor();
				} catch (InterruptedException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();	
				}

	            /*Process proc = Runtime.getRuntime().exec("/opt/iproject/videolabelling/asrlabels/transcribemany.sh"+" "+url);
	  
	            BufferedReader read = new BufferedReader(new InputStreamReader(
	                    proc.getInputStream()));
	            try {
	                proc.waitFor(10, TimeUnit.SECONDS);
	            } catch (InterruptedException e) {
	                System.out.println(e.getMessage());
	            }
	            StringBuilder output = new StringBuilder();
	            String line = "";
	            int i = 0;
	            while (read.ready() ) {


		            	//line = read.readLine();
		            	if (line.length() == 0) {
		    	            try {
		    	                proc.waitFor(10, TimeUnit.SECONDS);
		    	            } catch (InterruptedException e) {
		    	                System.out.println(e.getMessage());
		    	            }
		            		
		            	}
		            		
		            	output.append(line + "\n");
	            		System.out.println(line);
	            		i = i+1;

	            }*/
	            String line = "";
	            int i = 0;
	            Process outproc = Runtime.getRuntime().exec("tail -f /opt/iproject/videolabelling/asrlabels/asrmany");
	      	  
	            BufferedReader outread = new BufferedReader(new InputStreamReader(
	            		outproc.getInputStream()));
	            StringBuilder finaloutput = new StringBuilder();
	            try {
	            	outproc.waitFor(10, TimeUnit.SECONDS);
	            } catch (InterruptedException e) {
	                System.out.println(e.getMessage());
	            }
	            while (outread.ready()) {
	            	line = outread.readLine();
	            	finaloutput.append(line + "\n");
	            	if (i > 100) {
	            		break;
	            	}
	            	else {
	            		System.out.println(i);
	            		System.out.println(line);
	            		i = i+1;
	            	}
	            }
	            return finaloutput.toString();
	        }
	        catch (IOException e) {
	            e.printStackTrace();
	        }
	        return "ERROR in shell execution";
	    }
	    
	    public  void runTranscribeVideo(String url)
	    {
	    	System.out.println("runTranscribeVideo" + "control comes inside runTranscribeVideo"); 
	        try {
	  
	            System.out.println(
	                System.getProperty("os.name"));
	            System.out.println();
	            
	            System.out.println("Command = "+"/opt/iproject/videolabelling/asrlabels/transcribemany.sh"+" "+url);
	           /*try {
					Runtime.getRuntime().exec(new String[]{"/opt/iproject/videolabelling/asrlabels/transcribemany.sh","-c", url}).waitFor();
				} catch (InterruptedException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();	
				}*/
	            Process proc = Runtime.getRuntime().exec("/opt/iproject/videolabelling/asrlabels/transcribemany.sh"+" "+url);
	     	  
	            BufferedReader read = new BufferedReader(new InputStreamReader(
	                    proc.getInputStream()));
	            try {
	                proc.waitFor(10, TimeUnit.SECONDS);
	            } catch (InterruptedException e) {
	                System.out.println(e.getMessage());
	            }
	            StringBuilder output = new StringBuilder();
	            String line = "";
	            int i = 0;
	            while (read.ready() ) {


		            	line = read.readLine();
		            	if (line.length() == 0) {
		    	            try {
		    	                proc.waitFor(10, TimeUnit.SECONDS);
		    	                System.out.println("process waiting");
		    	            } catch (InterruptedException e) {
		    	            	System.out.println("exception in the wait block");
		    	                System.out.println(e.getMessage());
		    	            }
		            		
		            	}
		            		
		            	output.append(line + "\n");
	            		System.out.println(line);
	            		i = i+1;

	            }


	        }
	        catch (IOException e) {
	            e.printStackTrace();
	        }
	        System.out.println("runTranscribeVideo" + "control returns from runTranscribeVideo"); 
	    }
	    
	    public  String runGetTranscribedText(String url)
	    {
	    	System.out.println("runGetTranscribedText" + "control comes inside runGetTranscribedText"); 
                String filename = url.split("=")[1];
                System.out.println(filename);
                StringBuilder finaloutput = new StringBuilder();

	            String filefullpath = "/opt/iproject/videolabelling/asrlabels/"+filename +".txt";
	            boolean looper = true;
	            
	           int attempt = 5;
	            
	            while (looper) {
	
	            try {
	                File myObj = new File(filefullpath);
	                Scanner myReader = new Scanner(myObj);
	                while (myReader.hasNextLine()) {
	                  String data = myReader.nextLine();
	                  System.out.println(data);
	                  finaloutput.append(data+'\n');
	                }
	                myReader.close();
	                looper = false;
	              } catch (FileNotFoundException e) {
	                System.out.println("An error occurred.");
	                attempt = attempt - 1;
	                try {
	                    Thread.sleep(1000);
	                }
	                catch( InterruptedException ex ) {
	                	looper = false;
	                }
	                if (attempt == 0) {
	                	looper = false;
               
	                }
	                //e.printStackTrace();
	              }
	            }
	            System.out.println("runGetTranscribedText" + "control returns with error from runGetTranscribedText");
	            return finaloutput.toString();
	        }

	    
	    public  String runGetTranscribedCsv(String url)
	    {
	        try {
                String filename = url.split("=")[1];
                System.out.println(filename);

	            String filefullpath = "/opt/iproject/videolabelling/asrlabels/"+filename +".csv";
	
	            String line = "";
	            int i = 0;
	            Process outproc = Runtime.getRuntime().exec("tail -n100 "+	filefullpath);
	      	  
	            BufferedReader outread = new BufferedReader(new InputStreamReader(
	            		outproc.getInputStream()));
	            StringBuilder finaloutput = new StringBuilder();
	            try {
	            	outproc.waitFor(10, TimeUnit.SECONDS);
	            } catch (InterruptedException e) {
	                System.out.println(e.getMessage());
	            }
	            while (outread.ready()) {
	            	line = outread.readLine();
	            	finaloutput.append(line + "\n");
	            	if (i > 100) {
	            		break;
	            	}
	            	else {
	            		System.out.println(i);
	            		System.out.println(line);
	            		i = i+1;
	            	}
	            }
	            return finaloutput.toString();
	        }
	        catch (IOException e) {
	            e.printStackTrace();
	        }
	        return "ERROR in shell execution";
	    }
	    

}
