package com.video.tagger;

import org.springframework.data.jpa.repository.Query;

public interface IcontentRepository {
	@Query("select * from Student")
	String getStudent() ;
	

}
