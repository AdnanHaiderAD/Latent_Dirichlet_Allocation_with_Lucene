package latentDirichletAllocation;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermFreqVector;
import org.apache.lucene.search.Similarity;

import co.annotate.LatentSemanticAnalysis.AnalyseDocs;
import co.annotate.LatentSemanticAnalysis.AnalyticsSearch;
import co.annotate.LatentSemanticAnalysis.MatrixOperations;

/* @author Adnan  
 */

public class AnalyseWithLDA {
	private String dirpath;
	private WordStore store;
	
	public AnalyseWithLDA(String dirpath, int numberOfTopics){
		this.dirpath = dirpath;
		try {
			//===============================================================================================
			/** initialise a reader on the lucene directory**/
			IndexReader reader = AnalyseDocs.readLuceneIndex(dirpath);
			
			this.store = new WordStore(reader.terms());
			
			long time  = System.currentTimeMillis();
			System.out.println("document term occurrence matrix is being created");
			int[][]  documentByTermMatrix =createDocTermOccurrenceMatrix(reader);
			reader.close();
			reader = null;
			System.gc();
			System.out.println(" matrix created in " +(System.currentTimeMillis()-time));
			//===============================================================================================
		
			/** execute latent dirichlet allocation**/
			LDAGibbsPrototype lda = new LDAGibbsPrototype(documentByTermMatrix,numberOfTopics, store.numberofWords());
			time  = System.currentTimeMillis();
			System.out.println("latent dirichlet allocation is being performed");
			lda.gibbsSamplerstart();
			System.out.println("LDA done in  " +(System.currentTimeMillis()-time));
			
			float[][] docTopicMatrix = lda.get_thetaMatrix();
			visualiseDocTopicMatrix(docTopicMatrix);
			
			float[][] topicwordMatrix = lda.get_phiMatrix();
			visualisetopics(topicwordMatrix);
			
			
			/** visualise LDA representations of documents**/
			printLDADistanceMatrix(docTopicMatrix);
			==========================================================================
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}

	

	private int[][] createDocTermOccurrenceMatrix(IndexReader reader) {
		int docNum = reader.maxDoc();
		Similarity measure = Similarity.getDefault();
		
		//filter out deleted docs from lucene index
		ArrayList<Integer> listOfDocs = new ArrayList<Integer>();
		for(int i=0;i<docNum;i++){
			if (!reader.isDeleted(i)) {
				listOfDocs.add(i);
			}
		}
	    /** create document by term  matrix where each entry in a document is an instance of a term**/
		int[][] documentTermMatrix  = new int [listOfDocs.size()][];
		for (int j=0;j<listOfDocs.size();j++){
    		try{
    			TermFreqVector v =reader.getTermFreqVector(listOfDocs.get(j), "contents");
    			if( v == null) continue;
    			String[] terms = v.getTerms();
    			int[] termids = new int[terms.length];
    			for (int k = 0; k < terms.length; k++){
    					termids[k] = store.getwordID(terms[k]);
    			}
    			documentTermMatrix[j] = termids;
    			
    			
    		}	
    		catch (Exception e) {
 				System.out.println("doc: "+j+"");
 				e.printStackTrace();
 			}
    	}
		return documentTermMatrix;
	}
	
	//==================================================================================================================
	/** output the topic composition present in individual  documents and the word composition present in individual topics**/
	public void visualiseDocTopicMatrix(float[][] docTopicMatrix){
		for (int i =0; i < docTopicMatrix.length;i++){
			StringBuffer str = new StringBuffer();
			str.append("doc "+ Integer.toString(i)+ " =  ");
			for (int j =0 ; j < docTopicMatrix[i].length;j++ ){
				str.append(" topic " + Integer.toString(j) + " * " + Float.toString(docTopicMatrix[i][j]));
			}
			System.out.println(str.toString());
			
		}
	
	}
	
	public void  visualisetopics(float[][] topicWordMatrix){
		for (int i = 0; i < topicWordMatrix.length;i++){
			StringBuffer str = new StringBuffer();
			str.append(" topic " + Integer.toString(i) + " = " );
			int[] topicIndexes = new int[topicWordMatrix[i].length];
			float [] topicWorddist = topicWordMatrix[i];
			for (int k = 0; k <topicIndexes.length;k++) {
				topicIndexes[k] = k;
			}
			for (int k = 0; k < topicWorddist.length; k++){
				int max = k;
				for ( int l = k+1 ; l < topicWorddist.length; l++ ){
					if (topicWorddist[l] > topicWorddist[max]) max = l;
				}
				float tmp_value = topicWorddist[k];
				int tmp_index =topicIndexes[k] ;
				topicWorddist [k] = topicWorddist[max];
				topicIndexes[k] = topicIndexes[max];
				topicWorddist[max] = tmp_value;
				topicIndexes[max] = tmp_index;
			}
			int limit = topicIndexes.length>100 ? 100: topicIndexes.length;
			for (int k = 0; k <limit; k++){
				float value = topicWorddist[k];
				if (value > 0.00005){
					str.append(store.getWord(topicIndexes[k])+ "*" + Float.toString(value)+ " ");
				}
			}
			System.out.println(str.toString());
			System.out.println();
		}
	}
	
	private double[][] convertToDoublePrecision(float[][] matrix) {
		double[][] doubleMatrix = new double[matrix.length][matrix[0].length];
		for (int i = 0; i < matrix.length; i++){
			for (int j = 0 ; j < matrix[i].length; j++ ){
						doubleMatrix[i][j] =  matrix[i][j]; 
			}
		}
		return doubleMatrix;

	}
	//=========================================================================================================
	
	
	/** prints the distance scores between rows of the LDA generated doc by topic matrix**/
	public void printLDADistanceMatrix(float[][] doctopicMatrix ){
		System.out.println( "The similarity cost matrix generated by LDA is : ");
		//-----------------------------------------------------------------------------------
		// The computeEuclideandistanceMatrix function takes a double[][] input so need to type cast it
		double[][] docTopicMatrix = convertToDoublePrecision(doctopicMatrix);
		printDistanceMatrix(docTopicMatrix);
	}
	
	public void printDistanceMatrix(double[][] docTopicMatrix){
		double [][] result = MatrixOperations.computeEuclideandistanceMatrix(docTopicMatrix);
		for ( int i =0; i < result.length;i++){
			StringBuffer str = new StringBuffer();
				for (int j = 0; j< result.length; j++){
					str.append(result[i][j]!=0 ? MatrixOperations.roundToSignificantFigures(result[i][j],3) :0 );
					str.append(" ");
				}
				System.out.println(str.toString());
		}
	}
//========================================================================================================================	
	public static void main (String[] args){
		String path = "/home/adnan/annotestore/private/_ws/027/776/027776/AD/analytics/_luceneV36_wsa_refined";
		int topic = 4;
		new AnalyseWithLDA(path, topic);
	}

}
