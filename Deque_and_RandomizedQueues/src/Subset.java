/**
 * @author Lawrence Abadier
 * @version 1.0
 * @date 02/02/2015
 * 
 */

public class Subset {

   /**
    * Takes a command-line integer k; reads in a sequence of N strings from
    * standard input using StdIn.readString(); and prints out exactly k of them,
    * uniformly at random. Each item from the sequence can be printed out at
    * most once. You may assume that 0 <= k <= N, where N is the number of
    * string on standard input.
    * 
    * @param args
    *           is a sequence of N strings from standard input
    */
   public static void main(String[] args) {
      RandomizedQueue<String> randStr = new RandomizedQueue<String>();
      
      int k = Integer.parseInt(args[0]);
      
      while (!StdIn.isEmpty()) {
         String s = StdIn.readString();
         randStr.enqueue(s);
      }

      for (int i = 0; i < k; i++) {
         StdOut.print(randStr.dequeue());
      }

   }

}
