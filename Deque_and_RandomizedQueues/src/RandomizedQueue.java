import java.util.Iterator;  
import java.util.NoSuchElementException;  

/**
 * @author Lawrence Abadier
 * @version 1.0
 * @date 02/02/2015
 * 
 */
public class RandomizedQueue<Item> implements Iterable<Item> {
   private int size = 0;   // The starting size of the array
   private Item[] queue;  

   /**
    * Constructs an empty randomized queue
    */
   public RandomizedQueue() {
      queue = (Item[]) new Object[2];  
   }

   /**
    * Checks if the queue is empty
    * 
    * @return true if the queue is empty
    */
   public boolean isEmpty() {
      return size == 0;  
   }

   /**
    * Return the number of elements in the queue
    * 
    * @return the number of elements in the queue
    */
   public int size() {
      return size;  
   }

   /**
    * Adds an item to the front of the queue
    * 
    * @param item
    *           is added to the front of the queue
    */
   public void enqueue(Item item) {
      if (item == null)
         throw new java.lang.NullPointerException();  
      if (size == queue.length)
         resizeQueue(queue.length * 2);  
      queue[size++] = item;  
   }

   /**
    * Deletes and returns a random item in the queue
    * 
    * @return a random item from the queue
    */
   public Item dequeue() {
      if (isEmpty())
         throw new java.util.NoSuchElementException();  
      int i = StdRandom.uniform(size);   
      Item temp = queue[i];   
      queue[i] = queue[--size];    
      queue[size] = null;   // Set the last element of our queue to empty
      
      if (size > queue.length / 4)
         resizeQueue(queue.length / 2);   // Decrease the size of our array if it
                                        // is 1/4th full
      return temp;  
   }

   /**
    * Returns a random item from the queue
    * 
    * @return a random item from the queue
    */
   public Item sample() {
      if (isEmpty())
         throw new java.util.NoSuchElementException();  
      return queue[StdRandom.uniform(size)];  
   }

   /**
    * Return an independent iterator over items in random order
    * 
    * @return an independent iterator over items in random order
    */
   public Iterator<Item> iterator() {
      return new QueueIterator();  
   }

   /**
    * Our inner iterator class
    */
   private class QueueIterator implements Iterator<Item> {
      private int currentIndex;  
      private int[] randIndicies; 
      
      public QueueIterator(){
         randIndicies = new int [size]; 
         for(int i =0;  i < randIndicies.length;  i++){
            randIndicies[i] = i; 
         }
         StdRandom.shuffle(randIndicies); 
      }

      @Override
      /**
       * Checks if the next element is not null
       */
      public boolean hasNext() {
         return currentIndex < size;  
      }

      @Override
      /**
       * Iterates to a random element in the queue
       */
      public Item next() {
         //if current index is == to size we still want to return an index
         if (currentIndex > size || size == 0)
            throw new java.util.NoSuchElementException();  

         return queue[randIndicies[currentIndex++]];  
      }

      @Override
      /**
       * Unsupported method
       */
      public void remove() {
         throw new java.lang.UnsupportedOperationException();  
      }

   }

   private void resizeQueue(int newCapacity) {
      @SuppressWarnings("unchecked")
      Item[] newQueue = (Item[]) new Object[newCapacity];  
      int index = 0;  
      for (Item i : queue) {
         newQueue[index++] = i;  
      }
      queue = newQueue;  
   }
}
