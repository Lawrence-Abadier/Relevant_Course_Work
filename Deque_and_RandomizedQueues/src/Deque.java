/**
 * @author Lawrence Abadier
 * @version 1.0
 * @date 02/02/2015
 * 
 */

import java.util.Iterator; 


public class Deque<Item> implements Iterable<Item> {
   private int size;  // The number of elements in our deque
   private Node first; // The first element in our deque
   private Node last; // The last element in our deque

   // A helper doubly-linked-list class
   private class Node {
      private Item item; 
      private Node next; 
      private Node previous; 

      private Node() {
         next = null; 
         previous = null; 
      }
   }

   /**
    * Initializes an empty deque
    */
   public Deque() {
      size = 0; 
      first = null; 
      last = null; 

   }

   /**
    * Checks if our deque is empty
    * 
    * @return Checks if our deque is empty
    */
   public boolean isEmpty() {
      return first == null; 
   }

   /**
    * Returns the size of our deque
    * 
    * @return the size of our deque
    */
   public int size() {
      return size; 
   }

   /**
    * Inserts an item at the front of our deque
    * 
    * @param item
    *           is the item to be added at the front of the deque
    */
   public void addFirst(Item item) { // insert the item at the front
      if (item == null)
         throw new NullPointerException(); 
      Node newFirst = new Node(); 

      if (!isEmpty()) {
         newFirst.next = first; 
         first.previous = newFirst; 
      }

      first = newFirst; 
      first.item = item; 
      size++; 
      // if its the only node in the queue then first=last
      if (last == null)
         last = first; 
   }

   /**
    * Inserts an item at the end of our deque
    * 
    * @param item
    *           is the item to be added at the end of our deque
    */
   public void addLast(Item item) { // insert the item at the end
      if (item == null)
         throw new NullPointerException(); 
      Node newLast = new Node(); 

      if (last != null) {
         last.next = newLast; 
         newLast.previous = last; 
      }
      last = newLast; 
      last.item = item; 
      size++; 
      // if its the only node in the queue then first=last
      if (isEmpty())
         first = last; 
   }

   /**
    * Removes the first item in the deque and returns its item
    * 
    * @return the item from the deleted element
    */
   public Item removeFirst() { // delete and return the item at the front
      if (isEmpty())
         throw new java.util.NoSuchElementException(); 

      Item temp = first.item; 
      if (size == 1) {
         first = null; 
         last = null; 
      } else {
         first = first.next; 
         first.previous = null; 
      }

      size--; 

      return temp; 
   }

   /**
    * Removes the last item in the deque
    * 
    * @return java.util.NoSuchElementException() if there is no elements in the
    *         deque
    */
   public Item removeLast() { // delete and return the item at the end
      if (last == null)
         throw new java.util.NoSuchElementException(); 
      Item temp = last.item; 
      if (size == 1) {
         first = null;  
         last = null; 
      } else {
         last = last.previous; 
         last.next = null; 
      }

      size--; 

      return temp; 
   }

   /**
    * Returns a deque iterator
    * 
    * @return a deque iterator
    */
   public Iterator<Item> iterator() { // return an iterator over items in order
      // from front to end
      return new DequeIterator(); 
   }

   // an iterator, doesn't implement remove() since it's optional
   private class DequeIterator implements Iterator<Item> {
      private Node current = first; 

      public boolean hasNext() {
         return current != null; 
      }

      public void remove() {
         throw new UnsupportedOperationException(); 
      }

      public Item next() {
         if (!hasNext())
            throw new java.util.NoSuchElementException(); 
         Item item = current.item; 
         current = current.next; 
         return item; 
      }
   }

   public static void main(String[] args) { // unit testing
      Deque<Integer> test = new Deque<Integer>();

      for (int i = 0; i < 500; i++) {
         int rand = StdRandom.uniform(test.size());
         if (StdRandom.bernoulli())
            test.addFirst(rand);
         else
            test.addLast(rand);
      }
      
      test.removeLast();
      test.addFirst(2);
      test.addFirst(3);
      test.addFirst(4);
      test.addLast(20);
      test.addFirst(22);
      test.addLast(44);
      test.removeLast();
      test.removeFirst();

      while (test.iterator().hasNext()) {
         test.removeLast();
      }
      StdOut.println("Done");
   }
}