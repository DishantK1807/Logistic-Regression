import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;


public class LogisticRegression {
    
	public LogisticRegression(){
		
	}
	
	public int lines;
	public int columns;
	public float gradient[];
	public float Beta[];
	public int X[][];
	public int Y[];
    public int epochs;
    public float learningRate;
    public float z[];
	
	
	public void Training(String path, int e, float lR){
		
		try {
				BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
				epochs = e;
				learningRate = lR;
				
				columns= Integer.parseInt(reader.readLine());
				lines = Integer.parseInt(reader.readLine());
				
				X = new int [lines][columns +1];
				Y = new int [lines];
				
				
				
				for(int i=0; i<lines;i++){
							
				    String InputLines = reader.readLine();
							
					String Inputs[] = InputLines.split(":");
					String XValues[] = Inputs[0].split(" ");
					Inputs[1] = Inputs[1].trim();
					int YValue = Integer.parseInt(Inputs[1]);
					Y[i] = YValue;
					
					X[i][0]=1;
					
					for(int c=0; c<columns;c++){
						X[i][c+1] = Integer.parseInt(XValues[c]);
					}
				}
				
				Gradientcalc();

		}
		catch(Exception exp){
			exp.printStackTrace();
		}

	}
	
	public void Gradientcalc(){
		gradient = new float [columns + 1];
		Beta = new float [columns + 1];
		
		for(int c=0; c<=columns; c++){
			Beta[c] = 0;
		}
		
		int e;
		for (e=0; e< epochs; e++){
			for(int a=0; a<= columns; a++){
				gradient[a]= 0;
			}
			
			z = new float [lines];
			for (int l=0; l<lines; l++){
			   for (int m=0; m<=columns;m++){
				   z[l] += Beta[m]*X[l][m];
			   }
			}
		    
			for(int k=0; k<=columns; k++){
				for(int i =0; i<lines; i++){
					float func;
					func = (float) (1 / (1 + Math.pow(Math.E, -z[i])));
					gradient[k] += X[i][k]*(Y[i] - func);
				}
			}
			
			//
			for(int m=0; m<=columns; m++){
				Beta[m]+= learningRate*gradient[m];
			}
		}
	}
	
	public void Predicting(String path){
		try{
            BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
			
			reader.readLine();
			int predictLines = Integer.parseInt(reader.readLine());
			
			int i = 0;
			float z1;
			int goodpredict =0;
			float percentgood;
			while(i< predictLines){
            String InputLines = reader.readLine();
				
				String Inputs[] = InputLines.split(":");
				String XValues[] = Inputs[0].split(" ");
				Inputs[1] = Inputs[1].trim();
				int YValue = Integer.parseInt(Inputs[1]);
				int Y1 = YValue;
				
//				int X1[predictLines][columns +1];
//				X1[i][0]=1;
				z1 = 0;
				
//				for(int c=0; c<columns;c++){
//					X[i][c+1] = Integer.parseInt(XValues[c]);
//				}
				z1 += Beta[0];
				for (int m=0; m<columns;m++){
					z1 += Beta[m+1]*Integer.parseInt(XValues[m]);
				}
				
				float probability;
				probability = (float) (1 / (1 + Math.pow(Math.E, -z1)));
				
				if(probability>0.5){
					if(Y1==1){
						goodpredict++;
					}
				}
				else if(probability<=0.5){
					if(Y1==0){
						goodpredict++;
					}
				}
					
				i++;
			}
			percentgood = 100*(((float)goodpredict)/predictLines);
			System.out.println("Percentage of good predictions are:");
			System.out.println(percentgood);
		}
		catch(Exception exp){
			exp.printStackTrace();
		}
	}
	
	
	public static void main (String[] args){
		LogisticRegression test = new LogisticRegression();
		
		test.Training("InputFiles\\simple-train.txt",10000,0.0001f);
		test.Predicting("InputFiles\\simple-test.txt");
		
		test.Training("InputFiles\\vote-train.txt",10000,0.0001f);
		test.Predicting("InputFiles\\vote-test.txt");
		
		test.Training("InputFiles\\heart-train.txt",10000,0.0005f);
		test.Predicting("InputFiles\\heart-test.txt");
		
		test.Training("InputFiles\\heart-train.txt",10000,0.00002f);
		test.Predicting("InputFiles\\heart-test.txt");
		
		test.Training("InputFiles\\heart-train.txt",10000,0.00001f);
		test.Predicting("InputFiles\\heart-test.txt");
		
		test.Training("InputFiles\\heart-train.txt",10000,0.000002f);
		test.Predicting("InputFiles\\heart-test.txt");
		
		test.Training("InputFiles\\heart-train.txt",10000,0.000001f);
		test.Predicting("InputFiles\\heart-test.txt");
		
		test.Training("InputFiles\\heart-train.txt",10000,0.0000002f);
		test.Predicting("InputFiles\\heart-test.txt");
		
		test.Training("InputFiles\\heart-train.txt",10000,0.0000001f);
		test.Predicting("InputFiles\\heart-test.txt");
		
		
		
		

	}
}


	
