package co.annotate.latentDirichletAllocation;

import java.io.IOException;
import java.util.HashMap;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermEnum;

public class WordStore {

	HashMap<String,Integer> store ;
	HashMap<Integer,String> lookUp;
	int counter;
	
	public WordStore(TermEnum termIterator){
		store = new HashMap<String,Integer>();
		lookUp = new HashMap<Integer,String>();
		counter = 0;
		try {
			while(termIterator.next()){
				Term t = termIterator.term();
				 if (t.field().equals("contents") ){
					 insertWord(t.text());
				 }
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	public void insertWord(String word){
		store.put(word, counter );
		lookUp.put(counter, word);
		counter++;
	}
	
	public Integer getwordID(String word){
		return store.get(word);
	}
	
	public String getWord(Integer id){
		return lookUp.get(id);
	}
	
	public int numberofWords(){
		return counter;
	}
}
