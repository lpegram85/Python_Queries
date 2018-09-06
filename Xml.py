#xml is tree structure and give you attributes you have to traverse the tree to get the data
#itd be great to know the structure of a tree
from lxml import etree

data_string = """
<Bookstore>
   <Book ISBN="ISBN-13:978-1599620787" Price="15.23" Weight="1.5">
      <Title>New York Deco</Title>
      <Authors>
         <Author Residence="New York City">
            <First_Name>Richard</First_Name>
            <Last_Name>Berenholtz</Last_Name>
         </Author>
      </Authors>
   </Book>
   <Book ISBN="ISBN-13:978-1579128562" Price="15.80">
      <Remark>Five Hundred Buildings of New York and over one million other books are available for Amazon Kindle.</Remark>
      <Title>Five Hundred Buildings of New York</Title>
      <Authors>
         <Author Residence="Beijing">
            <First_Name>Bill</First_Name>
            <Last_Name>Harris</Last_Name>
         </Author>
         <Author Residence="New York City">
            <First_Name>Jorge</First_Name>
            <Last_Name>Brockmann</Last_Name>
         </Author>
      </Authors>
   </Book>
</Bookstore>
"""
root = etree.XML(data_string)

print(root.tag,type(root.tag))

#pretty print prints out the tree structure
print(etree.tostring(root, pretty_print=True).decode("utf-8"))

#to extract over all the elements of a tree but iterate all the way through

#this is a depth first iterator meaning goes down the tree first then goes on to the next column, basically printing the nodes
for element in root.iter():
    print(element)

for child in root:  # who are the children of bookstore book at the beginning and book at the end
    print(child)

for child in root: #this gives the tags
    print(child.tag)

#find the elements and then look for the children and get the text, it will not go down further than that
for element in root.iter("Author"):
    print(element.find('First_Name').text,element.find('Last_Name').text)

#xpath root.finall finds all tags that match
for element in root.findall("Book/Title"):
    print(element.text)

#xpath root.finall fins all tags that match, literally is the entire path it has to be
for element in root.findall("Book/Authors/Author/Last_Name"):
    print(element.text)

#only those books that have a weight of 1.5 in your xpath
for element in root.findall('Book[@Weight="1.5"]/Authors/Author/First_Name'):
    print(element.text)


--for element in root.iter("Book"):
--    print(element.find('Remark').txt)

