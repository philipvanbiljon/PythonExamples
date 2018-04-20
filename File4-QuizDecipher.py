#! /usr/bin/env python3

import sys

def decode(input):
  array = input.split()
  list = []

  #SOLUTION DESCRIPTION
  '''
  I started out by inspecting the quiz page and discovered the scramble function which was clearly printing
  the ASCII values plus a of each character + a random value. However it was adding a different value every three
  characters. I set out trying to work out the first value (a) by guessing the first letter.
  I first assumed it would start with congratulations but this seemed wrong, similarily my guess of it beginning Dear was wrong too.
  I then turned to the final character, assuming it was a period. This caused the first few characters to be H - - l - -
  After ruling out i as being the second character I tested whether the first word could be hello which worked
  Finally I altered the code to work with any version of the cipher rather than hard coding the values of a, b, c as before
  (This obviously assumes the message is always the same - or at least always starts with Hello - which seems to be the case)
  '''

  a = int(array[0]) - 72 #Ascii code for H - as this is always the first letter of the solution paragraph.
  if(a <= 0): a += 256
  b = int(array[1]) - 101 #e
  if(b <= 0): b += 256
  c = int(array[2]) - 108 #l
  if(c <= 0): c += 256
  print('The values assigned to a, b and c in the scramble javascript function are as follows:')
  print('a: ' + str(a) + ' b: ' + str(b) + ' c: ' + str(c))

  for i in range(0, len(array)):
    newint = 0
    if (i%3 == 0):
      newint = int(array[i]) - a
    if (i%3 == 1):
      newint = int(array[i]) - b
    if (i%3 == 2):
      newint = int(array[i]) - c
    if(newint <= 0): newint += 256
    list.append(unichr(newint))
  print('Below is the deciphered text: ')
  print(''.join(list))


if __name__ == "__main__":
    var = raw_input("Please enter cipher: ")
    decode(var)
