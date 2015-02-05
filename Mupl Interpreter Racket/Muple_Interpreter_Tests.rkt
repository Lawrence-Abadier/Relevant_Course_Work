#lang racket
;; Programming Languages Homework5 Simple Test
;; Save this file to the same directory as your homework file
;; These are basic tests. Passing these tests does not guarantee that your code will pass the actual homework grader

;; Be sure to put your homework file in the same folder as this test file.
;; Uncomment the line below and change HOMEWORK_FILE to the name of your homework file.
(require "hw5.rkt")

(require rackunit)

(define tests
  (test-suite
   "Sample tests for Assignment 5"
   
   ;; check racketlist to mupllist with normal list
   (check-equal? (racketlist->mupllist (list (int 3) (int 4))) (apair (int 3) (apair (int 4) (aunit))) "racketlist->mupllist test")
   
   ;; check mupllist to racketlist with normal list
   (check-equal? (mupllist->racketlist (apair (int 3) (apair (int 4) (aunit)))) (list (int 3) (int 4)) "racketlist->mupllist test")

   ;; tests if ifgreater returns (int 2)
   (check-equal? (eval-exp (ifgreater (int 3) (int 4) (int 3) (int 2))) (int 2) "ifgreater test")
   (check-equal? (eval-exp (ifgreater (int 3) (int 4) (add (int 1) (int 1)) (add (int 2) (int 2)))) (int 4) "ifgreater test")
   
   ;; mlet test
   (check-equal? (eval-exp (mlet "x" (int 1) (add (int 5) (var "x")))) (int 6) "mlet test")
   
   ;; call test
   (check-equal? (eval-exp (call (closure '() (fun #f "x" (add (var "x") (int 7)))) (int 1))) (int 8) "call test")
   
   ;;snd test
   (check-equal? (eval-exp (snd (apair (int 1) (int 2)))) (int 2) "snd test")
   
   ;; isaunit test
   (check-equal? (eval-exp (isaunit (closure '() (fun #f "x" (aunit))))) (int 0) "isaunit test")
   
   ;; ifaunit test
   (check-equal? (eval-exp (ifaunit (int 1) (int 2) (int 3))) (int 3) "ifaunit test")
   
   ;; mlet* test
   (check-equal? (eval-exp (mlet* (list (cons "x" (int 10))) (var "x"))) (int 10) "mlet* test")
   
   ;; ifeq test
   (check-equal? (eval-exp (ifeq (int 1) (int 2) (int 3) (int 4))) (int 4) "ifeq test")
   
   ;; mupl-map test
   (check-equal? (eval-exp (call (call mupl-map (fun #f "x" (add (var "x") (int 7)))) (apair (int 1) (aunit)))) 
                (apair (int 8) (aunit)) "mupl-map test")
   
   ;; problems 1, 2, and 4 combined test
   (check-equal? (mupllist->racketlist
   (eval-exp (call (call mupl-mapAddN (int 7))
                  (racketlist->mupllist 
                  (list (int 3) (int 4) (int 9)))))) (list (int 10) (int 11) (int 16)) "combined test")
   
   ;Additional Tests
    (check-equal? (eval-exp (int 17)) (int 17) "(eval-exp (int 17))")
    (check-equal? (eval-under-env (var "x") (list (cons "x" (int 17)))) (int 17) "single-variable lookup")
    (check-equal? (eval-exp (add (int 1) (int 2))) (int 3) "adding two ints")
    (check-equal? (eval-exp (ifgreater (int 1) (int 0) (int 1) (int 0))) (int 1) "ifgreater: true case")
    (check-equal? (eval-exp (ifgreater (int 0) (int 0) (int 1) (int 0))) (int 0) "ifgreater: false case")
    (check-equal? (eval-exp (apair (add (int 1) (int 2)) (ifgreater (int 2) (int 1) (int 3) (int 0)))) (apair (int 3) (int 3)) "apair of adds")
    (check-equal? (eval-exp (aunit)) (aunit) "eval-exp aunit")
    (check-equal? (eval-exp (isaunit (aunit))) (int 1) "isaunit: true case")
    (check-equal? (eval-exp (isaunit (int 100))) (int 0) "isaunit: false case")
    (check-equal? (eval-exp (mlet "xs" (apair (int 1) (int 2)) (var "xs"))) (apair (int 1) (int 2)) "mlet with pair")
    (check-equal? (eval-exp (mlet "xs" (apair (int 1) (int 2)) (fst (var "xs")))) (int 1) "mlet + fst")
    (check-equal? (eval-exp (fun #f "x" (aunit))) (closure null (fun #f "x" (aunit))) "fun evaluates to closure")
    (check-equal? (eval-exp (call (fun #f "x" (add (var "x") (int 1))) (int 1))) (int 2) "call a fun struct")
   ))

(require rackunit/text-ui)
;; runs the test
(run-tests tests)
