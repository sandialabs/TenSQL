#ifndef HASHSTORE_HPP
#define HASHSTORE_HPP

#include <stddef.h> //For size_t
#include <variant> //For hash
#include <limits> //For numeric_limits
#include <algorithm> //For fill
#include <iostream>

template<typename KEY_T, typename VALUE_T>
class HashStore
{
    protected:
        static constexpr size_t LL_NULL = std::numeric_limits<size_t>::max();

        struct DataNode {
            // Doubley linked list for nodes with hash collisions
            size_t *h_prev, h_next;

            // Doubley linked list for tracking free and allocated nodes
            size_t v_prev, v_next;

            // The key data
            KEY_T key;

            // The value data
            VALUE_T value;

            DataNode() : h_prev(NULL), h_next(LL_NULL), v_prev(LL_NULL), v_next(LL_NULL), key(), value()
            {
            }

            DataNode(const KEY_T &key, const VALUE_T &value) : h_prev(NULL), h_next(LL_NULL), v_prev(LL_NULL), v_next(LL_NULL), key(key), value(value)
            {
            }

            ~DataNode() {
                //std::cout << "Deconstructing DataNode: " << &key << std::endl;
                h_prev = NULL;
                h_next = LL_NULL;
                v_prev = LL_NULL;
                v_next = LL_NULL;
                //std::cout << "Completed deconstructing DataNode: " << &key << std::endl;
            }
        };

        struct HashTable {
            public:
                // Size of the hash lookup table
                size_t table_size;

                // Number of entries stored in the HashTable
                size_t table_entries;

                // Maximum number of entrie that can be stored in the HashTable
                size_t alloc_size;

                // h_table is the actual hash table.  It is a table of integer
                // offsets into the nodes array that identify the memory
                // backing each node.
                size_t* h_table;

                // This is the actual data stored within each hash table entry.
                // It also contains internal linked lists for tracking free and
                // allocated nodes, as well as short lists for resolving hash
                // collisions.
                DataNode* nodes;

                // Offset of the head of the list of free nodes
                size_t free_list;

                // Offset of the head of the list of allocated nodes
                size_t alloc_list;

                HashTable(size_t table_size=1024*1024) :
                    table_size(table_size), table_entries(0), alloc_size(0), h_table(NULL), nodes(NULL), free_list(0), alloc_list(LL_NULL)
                {
                    //HashTable can be at most half full
                    alloc_size = table_size / 2 + 1;
                    h_table = new size_t[table_size]();
                    nodes = (DataNode*)calloc(alloc_size, sizeof(DataNode));
                    //nodes = new DataNode[alloc_size]();

                    //Clear the hash table
                    std::fill(&h_table[0], &h_table[table_size], LL_NULL);

                    //Initialize the empty DataNode's
                    for(size_t it = 0; it < alloc_size; it++)
                    {
                        nodes[it].h_next = LL_NULL;
                        nodes[it].h_prev = NULL;
                        nodes[it].v_next = it+1;
                        nodes[it].v_prev = it-1;
                    }
                    nodes[0].v_prev = LL_NULL;
                    nodes[alloc_size-1].v_next = LL_NULL;
                }

                ~HashTable() {
                    //std::cout << "Deallocating hash table entries" << std::endl;
                    while(table_entries > 0)
                    {
                      //std::cout << "Initiating deallocation of node at: " << &(nodes[alloc_list].key) << std::endl;
                      deallocate_node(alloc_list);
                    }
                    //std::cout << "Deallocating h_table" << std::endl;
                    delete[] h_table;
                    //std::cout << "Deallocating nodes" << std::endl;
                    free(nodes);
                    //std::cout << "Finished ~HashTable" << std::endl;
                }

                class Iterator
                {
                    protected:
                        struct HashTable* table;
                        size_t idx;

                    public:
                        using iterator_category = std::bidirectional_iterator_tag;
                        using different_type = ssize_t;
                        using value_type = DataNode;
                        using pointer = DataNode*;
                        using reference = DataNode&;

                        Iterator(struct HashTable* table, size_t idx)
                            : table(table), idx(idx)
                        {

                        }

                        reference operator*() const { return table->nodes[idx]; }
                        pointer operator->() { return &(table->nodes[idx]); }
                        Iterator& operator++() { idx = table->nodes[idx].v_next; return *this; }
                        Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }
                        friend bool operator== (const Iterator& a, const Iterator& b) { return a.table == b.table && a.idx == b.idx; };
                        friend bool operator!= (const Iterator& a, const Iterator& b) { return a.table != b.table || a.idx != b.idx; };
                        size_t getIdx() const { return idx; }
                };

            protected:
                size_t allocate_node(const KEY_T &key, const VALUE_T &value, size_t* h_prev)
                {
                    size_t it_ret = free_list;
                    if(it_ret != LL_NULL) {
                        //Remove node from free_list
                        free_list = nodes[it_ret].v_next;

                        //Initialize the data node
                        new (&nodes[it_ret]) DataNode(key, value);

                        //Add node to allocated list
                        nodes[it_ret].v_next = alloc_list;
                        nodes[it_ret].h_prev = h_prev;
                        if(alloc_list != LL_NULL)
                          nodes[alloc_list].v_prev = it_ret;
                        alloc_list = it_ret;

                        //Increment the count of allocated nodes
                        table_entries++;
                    }

                    return it_ret;
                }

                void deallocate_node(size_t it_del) {
                    DataNode* node = &nodes[it_del];

                    //Remove from hash list
                    if(node->h_next != LL_NULL)
                        nodes[node->h_next].h_prev = node->h_prev;
                    if(node->h_prev != NULL)
                        *(node->h_prev) = node->h_next;

                    //Remove from the allocated list
                    if(node->v_next != LL_NULL)
                        nodes[node->v_next].v_prev = node->v_prev;
                    if(node->v_prev != LL_NULL)
                        nodes[node->v_prev].v_next = node->v_next;
                    else
                        alloc_list = node->v_next;

                    //std::cout << "Deallocating node " << &(node->key) << std::endl;
                    node->~DataNode();

                    //Add the node to the free list
                    node->v_next = free_list;
                    free_list = it_del;

                    table_entries--;

                    //std::cout << "Completed deallocating node: " << &(node->key) << std::endl;
                }

            public:
                size_t* find(const KEY_T &key) {
                    size_t hash = std::hash<KEY_T>{}(key);
                    size_t* ins = &h_table[hash % table_size];
                    while(*ins != LL_NULL) {
                        DataNode& cur = nodes[*ins];
                        if(cur.key == key)
                            break;
                        else
                            ins = &(cur.h_next);
                    }

                    return ins;
                }

                bool put(const KEY_T &key, const VALUE_T &value) {
                    size_t* ins = find(key);

                    if(*ins != LL_NULL)
                        return true;
                    else {
                        *ins = allocate_node(key, value, ins);
                        return false;
                    }
                }

                DataNode* get(const KEY_T &key) {
                    const size_t* ptr_ret = find(key);
                    if(*ptr_ret != LL_NULL)
                      return at(*ptr_ret);
                    else
                      return NULL;
                }

                DataNode* setdefault(const KEY_T &key, const VALUE_T &value) {
                    size_t* ins = find(key);

                    if(*ins == LL_NULL)
                        *ins = allocate_node(key, value, ins);

                    return at(*ins);
                }

                DataNode* at(size_t it) {
                    return &(nodes[it]);
                }

                bool remove(const KEY_T &key) {
                    const size_t* ptr_del = find(key);
                    return remove_at(*ptr_del);
                }

                bool remove_at(size_t nidx_del) {
                    //std::cout << "Removing node: " << nidx_del << std::endl << std::flush;
                    if(nidx_del != LL_NULL) {
                        deallocate_node(nidx_del);
                        return true;
                    }else{
                        return false;
                    }
                }

                size_t get_some_node() {
                    // Get an allocated node
                    if(alloc_list < alloc_size)
                        return alloc_list;
                    else
                        return LL_NULL;
                }

                Iterator begin() { return Iterator(this, alloc_list); }
                Iterator end() { return Iterator(this, LL_NULL); }
        };

        //Create two hashtables.  One twice the size of the other.  This allows
        //us to ensure O(1) inserts without needing to copy data, although we
        //do have to re-initialize a new hashtable when one of them gets 50%
        //full.  
        HashTable *front, *back;

        //Create a new frontbuffer twice as large as the old frontbuffer.  Free
        //the backbuffer, if it exists, as it is assumed to be empty.
        void resize() {
            while((front->table_entries+1) * 2 > front->table_size) {
                /*std::cout << "Resizing requires moving nodes.";
                std::cout << " front->table_size=" << front->table_size;
                std::cout << " front->table_entries=" << front->table_entries;*/
                if(back) {
                    //Move any remaining nodes out of the frontbuffer.
                    //std::cout << " back->table_entries=" << back->table_entries;
                    while(back->table_entries > 0) {
                        size_t it_del = back->get_some_node();
                        DataNode* move = back->at(it_del);
                        front->put(move->key, move->value);
                        back->remove_at(it_del);
                    }
                    delete back;
                }
                //std::cout << " new_back->table_entries=" << front->table_entries << std::endl;
                back = front;
                if(back->table_size > 0)
                  front = new HashTable(back->table_size * 2);
                else
                  front = new HashTable(2);
            }
        }

    public:
        HashStore(size_t table_size=1024*1024) :
            front(new HashTable(table_size)), back(NULL)
        {

        }

        HashStore(const HashStore& copy) :
            front(new HashTable(copy.getSize()*2+1)), back(NULL)
        {
            /*std::cout << "Copying HashStore: " << front->table_size;
            if(copy.front)
                std::cout << " " << copy.front->table_entries;
            if(copy.back)
                std::cout << " " << copy.back->table_entries;
            std::cout << std::endl;*/

            if(copy.back) {
                for(auto it_entry = copy.back->begin(); it_entry != copy.back->end(); it_entry++)
                    this->put(it_entry->key, it_entry->value);
            }

            for(auto it_entry = copy.front->begin(); it_entry != copy.front->end(); it_entry++)
                this->put(it_entry->key, it_entry->value);
        }

        static HashStore* copy(const HashStore& other)
        {
            //std::cout << "Copying" << std::endl;
            return new HashStore(other);
        }

        ~HashStore() 
        {
            //std::cout << "~HashStore() " << this << std::endl;
            delete front;
            if(back) delete back;
        }

        // Put a new key/value pair
        bool put(const KEY_T &key, const VALUE_T &value)
        {
            resize();

            if(back) {
                if(back->table_entries > 0) {
                  // Move a random node
                  size_t it_del = back->get_some_node();
                  DataNode* move = back->at(it_del);
                  front->put(move->key, move->value);
                  back->remove_at(it_del);

                  // Delete the node with the referenced key
                  it_del = back->remove(key);
                }

                if(back->table_entries == 0) {
                  delete back;
                  back = NULL;
                }
            }

            return front->put(key, value);
        }

        // Get the value associated with the key (or empty)
        VALUE_T* get(const KEY_T &key) {
            DataNode* node = NULL;

            if(back)
                node = back->get(key);
            if(node == NULL)
                node = front->get(key);

            if(node) {
                return &(node->value);
            }else{
                return NULL;
            }
        }

        // Get the value associated with the key (or empty)
        VALUE_T* setdefault(const KEY_T &key, const VALUE_T &value) {
            DataNode* node = NULL;

            if(back)
                node = back->get(key);
            if(node == NULL)
                node = front->setdefault(key, value);

            return &(node->value);
        }

        // Remove the key/value pair from the hash store (if it exists)
        bool remove(const KEY_T &key) {
            if(back)
                return back->remove(key);

            return front->remove(key);
        }

        size_t getSize() const {
            size_t ret = front->table_entries;
            if(back)
                ret += back->table_entries;
            return ret;
        }

        size_t toArrays(KEY_T** out_keys, VALUE_T** out_values, size_t max_items) {
            size_t num_copied = 0;
            if(back) {
                for(auto it_entry = back->begin(); num_copied < max_items && it_entry != back->end(); it_entry++, num_copied++)
                {
                    out_keys[num_copied] = &(it_entry->key);
                    out_values[num_copied] = &(it_entry->value);
                }
            }

            for(auto it_entry = front->begin(); num_copied < max_items && it_entry != front->end(); it_entry++, num_copied++)
            {
                out_keys[num_copied] = &(it_entry->key);
                out_values[num_copied] = &(it_entry->value);
            }

            return num_copied;
        }
};

#endif
