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

    fps = Bitset(1024)
    fps.set(0)
    fps.set(64)
    # b.set()
    # print(b)
    print(fps.to_int_arr())

