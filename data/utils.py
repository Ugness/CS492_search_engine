import pickle
from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_end_of_word = False

class Trie:
    def __init__(self, tokenizer):
        self.root = TrieNode()
        self.tokenizer = tokenizer

    def insert(self, word):
        node = self.root
        for char in word:
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def get_valid_tokens(self, node):
        valid_tokens = []
        if node.is_end_of_word:
            return [] # nothing valid.
        for char, child_node in node.children.items():
            valid_tokens.append(char)
        return valid_tokens
    
    def count_nodes(self, node=None):
        if node is None:
            node = self.root
        count = 1
        for child_node in node.children.values():
            count += self.count_nodes(child_node)
        return count

    def __len__(self):
        count = 1
        for child_node in self.root.children.values():
            count += self.count_nodes(child_node)
        return count

    def depth(self, node=None):
        if node is None:
            node = self.root
        if not node.children:
            return 0
        return 1 + max(self.depth(child_node) for child_node in node.children.values())


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def visualize_trie(trie):
        fig, ax = plt.subplots()
        ax.set_axis_off()

        def add_node(node, x, y, parent_x, parent_y, char):
            if node.is_end_of_word:
                circle = plt.Circle((x, y), radius=0.2, fill=False)
                ax.add_artist(circle)
                ax.text(x+0.2, y+0.2, char, ha='center', va='center')
            else:
                circle = plt.Circle((x, y), radius=0.05, fill=True)
                ax.add_artist(circle)
                ax.text(x+0.2, y+0.2, char, ha='center', va='center')
            if parent_x is not None and parent_y is not None:
                ax.plot([parent_x, x], [parent_y, y], 'k-')
            child_x = x - (len(node.children) - 1) / 2
            for child_char, child_node in node.children.items():
                add_node(child_node, child_x, y-1, x, y, child_char)
                child_x += 1

        add_node(trie.root, 0, 0, None, None, '')
        plt.savefig('sample.png')

    # Test the Trie class
    with open('dataset.pkl', 'rb') as f:
        data = pickle.load(f)
        _, database = data

    trie = Trie()
    i = 0
    for key in database.keys():
        i += 1
        trie.insert(key)
        print(key)
        if i > 10:
            break

    # Visualize the prefix tree
    visualize_trie(trie)