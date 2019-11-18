/*
 * SEEDChangeDetector.java
 * author: David T.J. Huang - The University of Auckland
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * either express or implied. See the License for the specific
 * language governing permissions and limitations under the
 * License.
 * 
 */

/**
 * 
 * Drift detection method as published in:
 * </p>
 * David Tse Jung Huang, Yun Sing Koh, Gillian Dobbie, and Russel Pears: Detecting Volatility Shift in Data Streams. ICDM 2014: 863-868
 * </p>
 * Usage: {@link #setInput(double)}
 * </p>
 * 
 * @author David T.J. Huang - The University of Auckland
 * @version 1.0
 */

import py4j.GatewayServer;

import java.util.ArrayList;
import java.util.List;

import java.util.Random;

public class SEEDChangeDetector
{
    public SEEDWindow window;
    private double DELTA;
    private int blockSize;
    private int elementCount;

    /**
     * Default constructor: sets all required parameters to default values.
     */
    public SEEDChangeDetector()
    {
	this.DELTA = 0.05;
	this.blockSize = 32;
	this.window = new SEEDWindow(blockSize, 1, 1, 0.01, 0.8, 75);
    }
    
    /**
     * Constructor for changing delta and blocksize, other parameters set to default.
     * @param delta
     * @param blockSize Recommended value: 32
     */
    public SEEDChangeDetector(double delta, int blockSize)
    {
	this.DELTA = delta;
	this.blockSize = blockSize;
	this.window = new SEEDWindow(blockSize, 1, 1, 0.01, 0.8, 75);
    }

    /**
     * Constructor for all required parameters.
     * @param delta 
     * @param blockSize Recommended value: 32
     * @param epsilonPrime Recommended values between [0.0025, 0.01]
     * @param alpha Growth parameter - Recommended values between [0.2, 0.8]
     * @param compressionterm Recommended values between [50, 100]
     */
    public SEEDChangeDetector(double delta, int blockSize, double epsilonPrime, double alpha, int term)
    {
	this.DELTA = delta;
	this.blockSize = blockSize;
	this.window = new SEEDWindow(blockSize, 1, 1, epsilonPrime, alpha, term);
    }   

    /**
     * Main method for passing in input values and performing drift detection
     * 
     * @param inputValue instance from the stream
     * @return boolean value true or false signaling whether a drift has occurred
     */
    public boolean setInput(double inputValue)
    {
	SEEDBlock cursor;

	addElement(inputValue);

	if (elementCount % blockSize == 0 && window.getBlockCount() >= 2) // Drift Point Check
	{
	    boolean blnReduceWidth = true;

	    while (blnReduceWidth)
	    {
		blnReduceWidth = false;
		int n1 = 0;
		int n0 = window.getWidth();
		double u1 = 0;
		double u0 = window.getTotal();

		cursor = window.getTail();
		while (cursor.getPrevious() != null)
		{
		    n0 -= cursor.getItemCount();
		    n1 += cursor.getItemCount();
		    u0 -= cursor.getTotal();
		    u1 += cursor.getTotal();
		    double diff = Math.abs(u1 / n1 - (u0 / n0));

		    if (diff > getADWINBound(n0, n1))
		    {
			blnReduceWidth = true;
			window.setHead(cursor);

			while (cursor.getPrevious() != null)
			{
			    cursor = cursor.getPrevious();
			    window.setWidth(window.getWidth() - cursor.getItemCount());
			    window.setTotal(window.getTotal() - cursor.getTotal());
			    window.setVariance(window.getVariance() - cursor.getVariance());
			    window.setBlockCount(window.getBlockCount() - 1);
			}

			window.getHead().setPrevious(null);

			return true;
		    }
		    cursor = cursor.getPrevious();
		}
	    }
	}

	return false;
    }

    private double getADWINBound(double n0, double n1)
    {
	double n = n0 + n1;
	double dd = Math.log(2 * Math.log(n) / DELTA);
	double v = window.getVariance() / window.getWidth();
	double m = (1 / (n0)) + (1 / (n1));
	double epsilon = Math.sqrt(2 * m * v * dd) + (double) 2 / 3 * dd * m;

	return epsilon;
    }

    public void addElement(double value)
    {
	window.addTransaction(value);
	elementCount++;
    }

    public class SEEDBlock
    {
	private SEEDBlock next;
	private SEEDBlock previous;

	private int blockSize;
	private double total;
	private double variance;
	private int itemCount;

	public SEEDBlock(int blockSize)
	{
	    this.next = null;
	    this.previous = null;
	    this.blockSize = blockSize;

	    this.total = 0;
	    this.variance = 0;
	    this.itemCount = 0;
	}

	public SEEDBlock(SEEDBlock block)
	{
	    this.next = block.getNext();
	    this.previous = block.getPrevious();
	    this.blockSize = block.blockSize;

	    this.total = block.total;
	    this.variance = block.variance;
	    this.itemCount = block.itemCount;
	}

	public void setNext(SEEDBlock next)
	{
	    this.next = next;
	}

	public SEEDBlock getNext()
	{
	    return this.next;
	}

	public void setPrevious(SEEDBlock previous)
	{
	    this.previous = previous;
	}

	public SEEDBlock getPrevious()
	{
	    return this.previous;
	}

	public int getBlockSize()
	{
	    return blockSize;
	}

	public void setBlockSize(int blockSize)
	{
	    this.blockSize = blockSize;
	}

	public void add(double value)
	{
	    itemCount++;
	    total += value;
	}

	public boolean isFull()
	{
	    if (itemCount == blockSize)
	    {
		return true;
	    } else
	    {
		return false;
	    }
	}

	public double getMean()
	{
	    return this.total / this.itemCount;
	}

	public void setTotal(double value)
	{
	    this.total = value;
	}

	public double getTotal()
	{
	    return this.total;
	}

	public void setItemCount(int value)
	{
	    this.itemCount = value;
	}

	public int getItemCount()
	{
	    return this.itemCount;
	}

	public void setVariance(double value)
	{
	    this.variance = value;
	}

	public double getVariance()
	{
	    return this.variance;
	}

    }

    public class SEEDWindow
    {
	private SEEDBlock head;
	private SEEDBlock tail;

	private int blockSize;
	private int width;
	private double total;
	private double variance;
	private int blockCount;

	private int DECAY_MODE = 1;
	private final int LINEAR_DECAY = 1;
	private final int EXPONENTIAL_DECAY = 2;

	private int COMPRESSION_MODE = 1;
	private final int FIXED_TERM = 1;

	private int decayCompressionCount = 0;
	private int linearFixedTermSize = 50;

	private double epsilonPrime = 0.0;
	private double alpha = 0.0;

	public SEEDWindow(int blockSize)
	{
	    clear();
	    this.blockSize = blockSize;
	    addBlockToHead(new SEEDBlock(blockSize));
	}

	public SEEDWindow(int blockSize, int decayMode, int compressionMode,
		double epsilonPrime, double alpha, int compressionTerm)
	{
	    clear();
	    this.blockSize = blockSize;
	    this.DECAY_MODE = decayMode;
	    this.COMPRESSION_MODE = compressionMode;
	    this.epsilonPrime = epsilonPrime;
	    this.alpha = alpha;
	    setCompressionTerm(compressionTerm);
	    addBlockToHead(new SEEDBlock(blockSize));
	}

	public void clear()
	{
	    head = null;
	    tail = null;
	    width = 0;
	    blockCount = 0;
	    total = 0;
	    variance = 0;
	}

	public void addTransaction(double value)
	{
	    if (tail.isFull())
	    {
		if (COMPRESSION_MODE == FIXED_TERM)
		{
		    if (tail.getPrevious() != null && decayCompressionCount > linearFixedTermSize)
		    {
			decayCompressionCount = 0;
			SEEDBlock cursor = tail;

			double epsilon = 0.0;

			int i = 0; 

			while (cursor != null && cursor.getPrevious() != null)
			{
			    double n0 = cursor.getItemCount();
			    double n1 = cursor.getPrevious().getItemCount();
			    double u0 = cursor.getTotal();
			    double u1 = cursor.getPrevious().getTotal();

			    double diff = Math.abs(u1 / n1 - (u0 / n0));

			    if (DECAY_MODE == LINEAR_DECAY)
			    {
				epsilon += epsilonPrime * alpha;
			    } 
			    else if (DECAY_MODE == EXPONENTIAL_DECAY)
			    {
				epsilon = epsilonPrime * Math.pow(1 + alpha, i);
			    }

			    if (diff < epsilon)
			    {
				compressBlock(cursor);
			    }
			    cursor = cursor.getPrevious();
			    i++; 
			}
		    }
		} 


		addBlockToTail(new SEEDBlock(this.blockSize));
		decayCompressionCount++;
	    }
	    tail.add(value);
	    total += value;

	    width++;
	    if (width >= 2)
	    {
		double incVariance = (width - 1) * (value - total / (width - 1))
			* (value - total / (width - 1)) / width;
		variance += incVariance;
		tail.setVariance(tail.getVariance() + incVariance);
	    }

	}

	public void compressBlock(SEEDBlock cursor)
	{
	    cursor.getPrevious().setTotal(
		    cursor.getTotal() + cursor.getPrevious().getTotal());
	    cursor.getPrevious().setItemCount(
		    cursor.getItemCount() + cursor.getPrevious().getItemCount());
	    cursor.getPrevious().setVariance(
		    cursor.getVariance() + cursor.getPrevious().getVariance());
	    cursor.getPrevious().setBlockSize(
		    cursor.getBlockSize() + cursor.getPrevious().getBlockSize());

	    if (cursor.getNext() != null)
	    {
		cursor.getPrevious().setNext(cursor.getNext());
		cursor.getNext().setPrevious(cursor.getPrevious());
	    } else
	    {
		cursor.getPrevious().setNext(null);
		tail = cursor.getPrevious();
	    }

	    blockCount--;
	}

	public boolean checkHomogeneity(SEEDBlock block)
	{
	    double diff = Math.abs(block.getMean() - block.getPrevious().getMean());
	    double epsilonPrime = getADWINBound(block.getItemCount(), block
		    .getPrevious().getItemCount());
	    // double epsilonPrime = 0.01;
	    if (diff < epsilonPrime)
	    {
		return true;
	    } else
	    {
		return false;
	    }
	}

	private double getADWINBound(double n0, double n1)
	{
	    double n = n0 + n1;
	    double dd = Math.log(2 * Math.log(n) / 0.99);
	    double v = variance / width;
	    double m = (1 / (n0)) + (1 / (n1));
	    double epsilon = Math.sqrt(2 * m * v * dd) + (double) 2 / 3 * dd * m;

	    return epsilon;
	}

	public void addBlockToHead(SEEDBlock block)
	{
	    if (head == null)
	    {
		head = block;
		tail = block;
	    } else
	    {
		block.setNext(head);
		head.setPrevious(block);
		head = block;
	    }
	    blockCount++;
	}

	public void removeBlock(SEEDBlock block)
	{
	    width -= block.getItemCount();
	    total -= block.getTotal();
	    variance -= block.getVariance();
	    blockCount--;

	    if (block.getPrevious() != null && block.getNext() != null)
	    {
		block.getPrevious().setNext(block.getNext());
		block.getNext().setPrevious(block.getPrevious());
		block.setNext(null);
		block.setPrevious(null);
	    } else if (block.getPrevious() == null && block.getNext() != null)
	    {
		block.getNext().setPrevious(null);
		head = block.getNext();
		block.setNext(null);
	    } else if (block.getPrevious() != null && block.getNext() == null)
	    {
		block.getPrevious().setNext(null);
		tail = block.getPrevious();
		block.setPrevious(null);
	    } else if (block.getPrevious() == null && block.getNext() == null)
	    {
		head = null;
		tail = null;
	    }
	}

	public void addBlockToTail(SEEDBlock block)
	{
	    if (tail == null)
	    {
		tail = block;
		head = block;
	    } else
	    {
		block.setPrevious(tail);
		tail.setNext(block);
		tail = block;
	    }
	    blockCount++;
	}

	public int getBlockCount()
	{
	    return this.blockCount;
	}

	public void setBlockCount(int value)
	{
	    this.blockCount = value;
	}

	public int getWidth()
	{
	    return this.width;
	}

	public void setWidth(int value)
	{
	    this.width = value;
	}

	public void setHead(SEEDBlock head)
	{
	    this.head = head;
	}

	public void setTail(SEEDBlock tail)
	{
	    this.tail = tail;
	}

	public SEEDBlock getHead()
	{
	    return this.head;
	}

	public SEEDBlock getTail()
	{
	    return this.tail;
	}

	public double getTotal()
	{
	    return this.total;
	}

	public void setTotal(double value)
	{
	    this.total = value;
	}

	public double getVariance()
	{
	    return this.variance;
	}

	public void setVariance(double value)
	{
	    this.variance = value;
	}

	public void setBlockSize(int value)
	{
	    if (value > 32)
	    {
		this.blockSize = value;
	    } else
	    {
		this.blockSize = 32;
	    }
	}

	public int getBlockSize()
	{
	    return this.blockSize;
	}

	public double getEpsilonPrime()
	{
	    return epsilonPrime;
	}

	public void setEpsilonPrime(double epsilonPrime)
	{
	    this.epsilonPrime = epsilonPrime;
	}

	public void setAlpha(double alpha)
	{
	    this.alpha = alpha;
	}

	public void setCompressionTerm(int value)
	{
	    this.linearFixedTermSize = value;
	}
    }
    
    public int test(
      int i
    ) {
      return i;
    }
    
    public int addition(int first, int second) {
      return first + second;
    }

    public List<Double> expr_seed_fpr(
      double delta, int blockSize,
      double epsilonPrime, double alpha,
      int term, 
      double mu, int length, 
      int n_repeat, int seed
      //int n_repeat, int n_trial
    ) {
        Random rnd = new Random(seed);
        List<Double> list = new ArrayList<>();
        for (int repeat = 0; repeat < n_repeat; ++repeat) {
          int count = 0;
          for (int i = 0; i < length; ++i) {
            double num = 0.0;
            if (rnd.nextDouble() <= mu) {
              num = 1.0;
            } 
            boolean res_i = setInput(num);
            if (res_i) {
              count++;
            }
          }
          list.add((double)count/n_repeat);
        }
        return list;
    }
    
    public List<Integer> expr_seed(
        double mu, 
        int length, 
        int seed
    ) {
        Random rnd = new Random(seed);
        List<Integer> list_trial = new ArrayList<>();
        for (int i = 0; i < length; ++i) {
            double num = 0.0;
            double r = rnd.nextDouble();
            if (r <= mu) {
              num = 1.0;
            } 
            boolean res_i = setInput(num);
            if (res_i) {
              list_trial.add(i);
            }
        }
        return list_trial;        
    }
    
 
    public List<Double> expr_volshift_fpr(
      double delta, int blockSize,
      double epsilonPrime, double alpha,
      int term, 
      double mu, int length, 
      int n_repeat, int seed
      //int n_repeat, int n_trial
    ) {
        Random rnd = new Random(seed);
        List<Double> list = new ArrayList<>();
        for (int repeat = 0; repeat < n_repeat; ++repeat) {
          int count = 0;
          for (int i = 0; i < length; ++i) {
            double num = 0.0;
            if (rnd.nextDouble() <= mu) {
              num = 1.0;
            } 
            boolean res_i = setInput(num);
            if (res_i) {
              count++;
            }
          }
          list.add((double)count/n_repeat);
        }
        return list;
    }
    
    public List<Double> generate_dataset_expr3(
      double mu1, double mu2, double sigma, 
      int length1, int length2,  
      int n_repeat1, int n_repeat2, 
      int seed
    ) {
      Random rnd = new Random(seed);
      List<Double> list_trial = new ArrayList<>();
    
      for (int repeat=0; repeat < n_repeat1; ++repeat) {
          for (int i=0; i<length1; ++i) {
              double x = mu1 + sigma * rnd.nextGaussian();
              list_trial.add(x);
          }
          for (int i=0; i<length1; ++i) {
              double x = mu2 + sigma * rnd.nextGaussian();
              list_trial.add(x);
          }
      }
      for (int repeat=0; repeat < n_repeat2; ++repeat){
          for (int i=0; i<length2; ++i) {
              double x = mu2 - (double)i/length2 + sigma * rnd.nextGaussian();
              list_trial.add(x);
          }
          for (int i=0; i<length2; ++i) {
              double x = mu1 + sigma * rnd.nextGaussian();
              list_trial.add(x);
          }
          for (int i=0; i<length2; ++i) {
              double x = (double)i/length2 + sigma * rnd.nextGaussian();
              list_trial.add(x);
          }
          for (int i=0; i<length2; ++i) {
              double x = mu2 + sigma * rnd.nextGaussian();
              list_trial.add(x);
          }
      }
        return list_trial;
    }
    
    
     public List<Integer> expr3_metachange_along_time(
      double mu1, double mu2, double sigma, 
      int length1, int length2,
      int n_repeat1, int n_repeat2, int seed
      //int n_repeat, int n_trial
    ) {
        Random rnd = new Random(seed);
        List<Integer> list_trial = new ArrayList<>();
        for (int repeat=0; repeat < n_repeat1; ++repeat) {
          for (int i=0; i<length1; ++i) {
              double x = mu1 + sigma * rnd.nextGaussian();
              boolean res_i = setInput(x);
              if (res_i) {
                  list_trial.add(2*length1*repeat + i);  
              }
          }
          for (int i=0; i<length1; ++i) {
              double x = mu2 + sigma * rnd.nextGaussian();
              boolean res_i = setInput(x);
              if (res_i) {
                  list_trial.add(2*length1*repeat+length1 + i);
              }
          }
      	}
	for (int repeat=0; repeat < n_repeat2; ++repeat){
          for (int i=0; i<length2; ++i) {
              double x = mu2 - (double)i/length2 + sigma * rnd.nextGaussian();
              boolean res_i = setInput(x);
              if (res_i) {
                  list_trial.add(2*length1*n_repeat1+repeat*length2+i);
              }
          }
          for (int i=0; i<length2; ++i) {
              double x = mu1 + sigma * rnd.nextGaussian();
              boolean res_i = setInput(x);
              if (res_i) {
                  list_trial.add(2*length1*n_repeat1+repeat*length2+length2+i);
              }
          }
          for (int i=0; i<length2; ++i) {
              double x = (double)i/length2 + sigma * rnd.nextGaussian();
              boolean res_i = setInput(x);
              if (res_i) {
                  list_trial.add(2*length1*n_repeat1+repeat*length2+2*length2+i);
              }
          }
          for (int i=0; i<length2; ++i) {
              double x = mu2 + sigma * rnd.nextGaussian();
              boolean res_i = setInput(x);
              if (res_i) {
                  list_trial.add(2*length1*n_repeat1+repeat*length2+3*length2+i);
              }
          }
        }
        return list_trial;
    }
   
    public List<Integer> expr_volshift(
      double mu1, double mu2,
      int length1, int length2,
      int n_repeat, int seed
    ) {
        Random rnd = new Random(seed);
        List<Integer> list_trial = new ArrayList<>();
        for (int repeat = 0; repeat < n_repeat; ++repeat) {
          for (int i = 0; i < length1; ++i) {
            double num1 = 0.0;
            double r = rnd.nextDouble();
            if (r <= mu1) {
              num1 = 1.0;
            } 
            boolean res_i = setInput(num1);
            if (res_i) {
              list_trial.add(repeat*2*length1+i);
            }
          }
          for (int i = 0; i < length1; ++i) {
            double num2 = 0.0;
            double r = rnd.nextDouble();
            if (r <= mu2) {
              num2 = 1.0;
            } 
            boolean res_i = setInput(num2);
            if (res_i) {
              list_trial.add(repeat*2*length1+length1+i);
            }
          }
        }
        for (int repeat = 0; repeat < n_repeat; ++repeat) {
          for (int i = 0; i < length2; ++i) {
            double num1 = 0.0;
            double r = rnd.nextDouble();
            if (r <= mu1) {
              num1 = 1.0;
            } 
            boolean res_i = setInput(num1);
            if (res_i) {
              list_trial.add(2*length1*n_repeat+2*length2*repeat+i);
            }
          }
          for (int i = 0; i < length2; ++i) {
            double num2 = 0.0;
            double r = rnd.nextDouble();
            if (r <= mu2) {
              num2 = 1.0;
            } 
            boolean res_i = setInput(num2);
            if (res_i) {
              list_trial.add(2*length1*n_repeat+2*length2*repeat+length2+i);
            }
          }
        }
      return list_trial;
    }

    public static void main(String args[]) {
      //double delta = Double.parseDouble(args[0]);
      double delta = 0.05;
      //int blockSize = Integer.parseInt(args[1]);
      int blockSize = 32;
      //double epsilonPrime = Double.parseDouble(args[2]);
      double epsilonPrime = 0.0075;
      //double alpha = Double.parseDouble(args[3]);
      double alpha = 0.6;
      //int term = Integer.parseInt(args[4]);
      int term = 75;

      SEEDChangeDetector detector = new SEEDChangeDetector(delta, blockSize, epsilonPrime, alpha, term);
      GatewayServer server = new GatewayServer(detector);
      server.start();
      //System.out.println("Gateway Server Started");
    }
}

