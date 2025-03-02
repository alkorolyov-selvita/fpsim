from bitset import Bitset

if __name__ == '__main__':
    b = Bitset(8)
    b.set(1)
    b.set(5)

    a = Bitset(8)
    a.set(1)

    print(a)
    print(b)
    print(a or b)
    print(a and b)

    b = Bitset(128)
    b.set(0)
    b.set(64)
    # b.set()
    print(b)
    print(b.to_int_arr())

