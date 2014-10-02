package co.annotate.latentDirichletAllocation;

public class LDAGibbsPrototype {
	//===========================================================================================================
		/** document by word matrix**/
		private int[][]  documents;
		
		/** document by topic-instance matrix -this has the same dimensions as the document by word matrix but records 
		 * the topic associated with each instance of the word**/ 
		private int[][] z;
		
		
		private int[][] doc_topicCountMatrix;
		
		private int[][] word_topicCountMatrix;
		
		private int numofstats;
		
		/** the distribution of topics in each doc**/
		private float[][] thetatsum;
		
		/** the distribution of words in each topic**/
		private float[][] phiSum;
		
		/** vocabulary size**/
		private int v;
		
		/** number of topics**/
		private int k;
		
		/** if the topic proportion prior αlpha receives is relatively large, then many topics will 
		 * be activated per document. On the other hand if αlpha  is small (say 0.1), then only few 
		 * topics will be activated per document. If αlpha is almost  zero, each document would have only one topic
		 */
		private double alpha;
		
		
		/** The value of β thus affects the granularity of the model: a corpus of documents can be sensibly factorized 
		 * into a set of topics at several different scales, and the particular scale assessed by the model will be set by 
		 * β. With scientific documents, a large value of β would lead the model to find a relatively small number of topics,
		 *  perhaps at the level of scientific disciplines, whereas smaller values of β will produce more topics that address 
		 *  specific areas of research.
		 */
		private double beta;
		
		/** stores the total number of words  associated with each topic*/
		private int [] topics_dist;
		/** stores the numbers of words associated with each doc*/
		private int[] docs_dist;
		
	//===================================================	
		/**
	     * burn-in period
	     */
	    private static int BURN_IN = 100;

	    /**
	     * max iterations
	     */
	    private static int ITERATIONS = 3000;

	    /**
	     * sample lag (if -1 only one sample taken)
	     */
	    private static int SAMPLE_LAG;

	  //===========================================================================================================
	    public LDAGibbsPrototype(int[][] documents, int k,int v,int iterations,int sampleLag, int burnin){
	    	
	    	this.documents= documents;
	    	this.k = k;
	    	this.v = v;;
	    	topics_dist = new int [k];
	    	docs_dist = new int[documents.length];
	    	
	    	/**parameters**/
	    	this.alpha = 0.00001;
	    	this.beta = 0.81;
	    	
	    	this.ITERATIONS = iterations;
	    	this.BURN_IN = burnin ;
	    	this.SAMPLE_LAG = sampleLag;
	    }
	    
	    public LDAGibbsPrototype( int[][] documents, int k, int v){
	    	this(documents,k,v,50000,1000,10000);
	    }
	    
	   
	    public void gibbsSamplerstart(){
	    	/** initialise the state of the Markov Chain**/
	    	initialiseState();
	    	for (int i =0; i < ITERATIONS;i++){
	    		for (int m = 0; m< documents.length;m++){
	    			for (int n = 0; n < z[m].length ; n ++){
	    				int topic  = samplefromConditional(m,n);
	    				z[m][n]= topic;
	    			}
	    		}
	    		if ((i > BURN_IN) && (SAMPLE_LAG > 0) && (i % SAMPLE_LAG == 0)) {
	    			if(updateParameters(i)){
	    				System.out.println("Number of iterations needed to converge "+i);
	    				break;
	    			}
	    			
	    		}
	    		
	    	}
	    	
	    }
	    private void initialiseState(){
	    	int M = documents.length;
	    	this.numofstats = 0;
	    	
	    	z= new int[M][];
	    	doc_topicCountMatrix = new int[M][k];
	    	word_topicCountMatrix = new int [v][k];
	    	
	    	for (int m =0;m < M;m++){
	    		z[m] = new int[documents[m].length];
	    		
	    		for (int n=0; n<z[m].length;n++){
	    			int topic = (int) Math.floor(Math.random()*k);
	    			z[m][n]=topic;
	    			doc_topicCountMatrix[m][topic]++;
	    			word_topicCountMatrix[documents[m][n]][topic]++;
	    			topics_dist[topic]++;
	    		}
	    		docs_dist[m]= z[m].length;
	    	}
	    	
	    }
	    

		private int samplefromConditional(int m,int n) {
			int topic  = z[m][n];
			
			doc_topicCountMatrix[m][topic]--;
			word_topicCountMatrix[documents[m][n]][topic]--;
			topics_dist[topic]--;
		
			
			double [] p = new double [k];
			/**sampling zi from p(zi|z_i)**/
			for (int i = 0; i < k;i++){
				p[i] = (word_topicCountMatrix[documents[m][n]][i] + beta)/(topics_dist[i]+ v * beta) *(doc_topicCountMatrix[m][i]+ alpha)/(docs_dist[m]+ k*alpha);
			}
			
			 // cumulate multinomial parameters
	        for (int i = 1; i < p.length; i++) {
	            p[i] += p[i - 1];
	        }
	        // scaled sample because of unnormalised p[]
	        double u = Math.random() * p[k - 1];
	        for (topic = 0; topic < p.length; topic++) {
	            if (u < p[topic])
	                break;
	        }
	        
	        doc_topicCountMatrix[m][topic]++;
	        word_topicCountMatrix[documents[m][n]][topic]++;
	        topics_dist[topic]++;
			
			return topic;
		}
	    
	    private boolean updateParameters(int iteration) {
	    	boolean complete = false;
	    	if (thetatsum == null && phiSum == null ){
	    		thetatsum = new float[documents.length][k];
	    		phiSum = new float [k][v];
	    	}
	    	if ( iteration == Math.floor(((float)ITERATIONS-1)/(float)SAMPLE_LAG) *SAMPLE_LAG){
	    		updatedocTopicParameters();
	    		complete  = true;
	        }else{
	        	if (numofstats == 0) averagedocTopicParameters();
	        	else{
	        		complete = checkMeanConvergence();
	        		if (!complete) averagedocTopicParameters();
	        		else updatedocTopicParameters();
	        	}	
	        }
	    	
	    	for(int i = 0 ; i < k; i++){
	    		for ( int n = 0; n < v; n++){
	    			phiSum[i][n] = (float)(word_topicCountMatrix[n][i]/(float)topics_dist[i]) ;
	    		}
	    	}
	    	return complete;
	    	
		}
	    private void updatedocTopicParameters(){
	    	for (int m = 0; m < documents.length; m++){
	    		for (int j =0; j< k; j++){
	    			thetatsum[m][j] = (float)(doc_topicCountMatrix[m][j]/(float)docs_dist[m]) ;
	    		}
	    	}
	    }
	    private void averagedocTopicParameters(){
	    	for (int m = 0; m < documents.length; m++){
	    		for (int j =0; j< k; j++){
	    			thetatsum[m][j] += (float)(doc_topicCountMatrix[m][j]/(float)docs_dist[m]) ;
	    		}
	    	}
	    	numofstats++;
	    }
	    private boolean checkMeanConvergence(){
	    	int count = 0;
	    	for (int m = 0; m < documents.length; m++){
	    		for (int j =0; j< k; j++){
	    			if (Math.abs(thetatsum[m][j]/numofstats - (float)(doc_topicCountMatrix[m][j]/(float)docs_dist[m])) <0.01){
	    				count++;
	    			};
	    		}
	    	}
	    	if ((float)count /(documents.length*k)>0.8){
	    		return true;
	    	}else{
	    		return false;
	    	}
	    		
	    }
	    
	    
	    public float[][] get_thetaMatrix(){
	    	return thetatsum;
	    }
	    
	    public float[][] get_phiMatrix(){
	    	return phiSum;
	    }
	    
	    
	    
	    
	}


