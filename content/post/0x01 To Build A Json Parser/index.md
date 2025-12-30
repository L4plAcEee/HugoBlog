---
date : '2025-12-30T16:12:08+08:00'
title : '【Re:C/C++】DIY之做一个JSON工具库'
summary : '基于 C++ 的JSON解析器\学习以及实现，然后基于此实现自己的JSON工具库'
---
## 1. 序言
### 1.1 课程相关资源
> 相关视频：【【C++项目实战】实现一个JSON解析器】 https://www.bilibili.com/video/BV1pa4y1g7v6/?share_source=copy_web&vd_source=b235e9c478ba07e2678b1ac01bb439c6  

> 代码仓库：https://github.com/L4plAcEee/my_json_parser

### 1.2 思考：基本思路
![alt text](image.png)


## 2. 课程代码精读
### 2.1 数据模型 JSON
```c++
struct JSONObject {
    std::variant<
    // - 类模板 std::variant 表示一个类型安全的 联合体。
    // - https://cppreference.cn/w/cpp/utility/variant

        std::monostate, 
        // - 单元类型，用作 std::variant 中行为良好的空替代, 在 JSONObject 数据模型中 充当 null 值。
        // - https://cppreference.cn/w/cpp/utility/variant/monostate

        bool, 
        int,  
        double, 
        std::string, 
        // - 基本数据类型：布尔值、整数、浮点数和字符串。

        std::vector<JSONObject>, 
        std::unordered_map<std::string, JSONObject>
        // - 复合数据类型：数组和对象。

        > inner;

    void do_print() const {
    // - 在 C++ 中，成员函数声明末尾的 const 关键字是一个非常重要的约束。它的核心作用是：承诺该函数不会修改对象的状态。
        // print(inner);
        printnl(inner);
    }

    template <class T>
    bool is() const {
        return std::holds_alternative<T>(inner);
    }

    template <class T>
    T const &get() const {
        return std::get<T>(inner);
    }

    template <class T>
    T &get() {
        return std::get<T>(inner);
    }
};
```

### 2.2 解析器部分
![alt text](image-1.png)
```c++
std::pair<JSONObject, size_t> parse(std::string_view json) {
    if (json.empty()) {
        return {JSONObject{std::monostate{}}, 0};
    } else if (size_t off = json.find_first_not_of(" \n\r\t\v\f\0"); off != 0 && off != json.npos){
        auto [obj, eaten] = parse(json.substr(off));
        return {std::move(obj), eaten + off};

    } else if ('0' <= json[0] && json[0] <= '9' || json[0] == '+' || json[0] == '-') {
        std::regex num_re{"[+-]?[0-9]+(\\.[0-9]*)?([eE][+-]?[0-9]+)?"};
        std::cmatch match;
        if (std::regex_search(json.data(), json.data() + json.size(), match, num_re)) {
            std::string str = match.str();
            if (auto num = try_parse_num<int>(str); num.has_value()) {
                return {JSONObject{*num}, str.size()};
            }

            if (auto num = try_parse_num<double>(str); num.has_value()) {
                return {JSONObject{*num}, str.size()};
            }
        }
        return {JSONObject{std::monostate{}}, 0};

    } else if (json[0] == '"') {
        std::string str;
        enum {
            Raw,
            Escaped,
        } phase = Raw;
        size_t i;
        for (i = 1;i < json.size(); ++i) {
            char ch = json[i];
            if (phase == Raw) {
                if (ch == '\\') {
                    phase = Escaped;
                } else if (ch == '"') {
                    i += 1;
                    break;
                } else {
                    str += ch;
                }
            } else if (phase == Escaped) {
                str += unescaped_char(ch);
                phase = Raw;
            }
        }
        return {JSONObject{std::move(str)}, i};

    } else if (json[0] == '[') {
        std::vector<JSONObject> res;
        size_t i;
        for (i = 1; i < json.size();) {
            if (json[i] == ']') {
                i += 1;
                break;
            } 
            auto [obj, eaten] = parse(json.substr(i));
            if (eaten == 0) {
                i = 0;
                break;
            }

            res.push_back(std::move(obj));
            i += eaten;

            if (json[i] == ',') {
                i += 1;
            }
        }
        return {JSONObject{std::move(res)}, i};
    
    } else if (json[0] == '{') {
        std::unordered_map<std::string, JSONObject> res;
        size_t i;
        for (i = 1; i < json.size();) {
            if (json[i] == '}') {
                i += 1;
                break;
            }
            auto [key_obj,key_eaten] = parse(json.substr(i));
            if (key_eaten == 0) {
                i = 0;
                break;
            }
            i += key_eaten;
            if (!std::holds_alternative<std::string>(key_obj.inner)) {
                // - 检查变体 v 是否持有替代类型 T。如果 T 在 Types... 中没有且仅出现一次，则此调用格式错误。
                // - https://cppreference.cn/w/cpp/utility/variant/holds_alternative
                i = 0;
                break;
            }

            if (json[i] == ':') {
                i += 1;
            }

            std:: string key = std::move(std::get<std::string>(key_obj.inner));
            auto [value_obj, value_eaten] = parse(json.substr(i));
            if (value_eaten == 0) {
                i = 0;
                break;
            }
            i += value_eaten;

            res.try_emplace(std::move(key), std::move(value_obj));
            if (json[i] == ',') {
                i += 1;
            }
        }  
        return {JSONObject{std::move(res)}, i};

    }

    return {JSONObject{std::monostate{}}, 0};
}
```

### 2.3 小技巧
#### 2.3.1 多态Visit + 递归lambda
> 递归 Lambda (Recursive Lambda) 配合 编译期分支 (if constexpr) 来处理递归数据结构。

- 解决递归 Lambda 的痛点：在 C++ 中，Lambda 无法直接通过名字递归调用自己（因为在 Lambda 定义完成前，名字还不可用）。采用了将 Lambda 自身作为第一个参数传入（auto &do_visit）的方案，这是目前最通用的递归 Lambda 实现方式。
  
- 编译期优化：使用了 if constexpr 结合 std::is_same_v。这确保了编译器只会在 school 确实是 JSONList（即 std::vector<JSONObject>）时才编译循环代码，避免了对非容器类型的非法操作，提升了编译速度和安全性。  
  
- 类型萃取准确：使用了 std::decay_t<decltype(school)>。由于 std::visit 传给 Lambda 的参数可能是引用或带 const 修饰，decay_t 能够将其还原为原始类型进行比对，非常严谨。
   

```c++
auto const &school = dict.at("school");
auto do_visit = [&] (auto &do_visit, JSONObject const &school) -> void {
    std::visit([&] (auto const &school) {
    //- std::visit 是一个用于访问 std::variant 中存储的值的函数模板。
    //  它接受一个访问者函数和一个或多个 std::variant 对象作为参数，
    //  并根据 std::variant 当前持有的类型调用访问者
    //- https://cppreference.cn/w/cpp/utility/variant/visit2
        if constexpr (std::is_same_v<std::decay_t<decltype(school)>, JSONList>) {
            for (auto const &subschool : school) {
                do_visit(do_visit, subschool);
            }
        } else  {
            print("I purchased my diploma from", school);
        }
    }, school.inner);
};
```

#### 2.3.2 Overload模式
> 这种模式在编译期就确定了调用关系，相比于 if (std::holds_alternative<...>) 这种运行时检查，效率更高且代码更简洁。  

```c++
template <class ...Fs>
struct overloaded : Fs... {
    using Fs::operator()...;
};

template <class ...Fs>
overloaded(Fs...) -> overloaded<Fs...>;

// ...

auto [obj, eaten] = parse(str);
print(obj);
std::visit(
    overloaded{
        [&] (int val) {
            print("int is:", val);
        },
        [&] (double val) {
            print("double is:", val);
        },
        [&] (std::string val) {
            print("string is:", val);
        },
        [&] (auto val) {
            print("unknown object is:", val);
        },
    },
    obj.inner);
```
测试输入：
```txt
3.2
double is: 3.2

3
int is: 3

"name"
string is: "name"
```

没问题，将这部分内容翻译为双语版不仅能提升笔记的格调，还能帮助你更精准地掌握相关的技术术语。

以下是为你优化后的双语版本：

---

## 3. TODO 与后日谈

### 3.1 Course Assignments

* [ ] **1. Special Literals Parsing | 特殊字面量解析** 
    Implement the parsing of `null`, `false`, and `true`, and verify their correctness in nested/recursive structures.
    实现 `null`、`false` 和 `true` 的解析，并验证它们在递归嵌套情况下的正确性。
* [ ] **2. Single-Quote String Support | 单引号字符串支持** 
    Extend the parser to support string literals enclosed in single quotes, e.g., both `'string'` and `"string"` should be valid.
    支持单引号定义的字符串字面量，例如 `'string'` 和 `"string"` 均应能正常工作。
* [ ] **3. Hex Escape Sequences | 十六进制转义序列** 
    Support hex character escape sequences such as `\x0D` (this may require introducing new state enums like `Hex1`, `Hex2`).
    支持字符串中的十六进制字符转义（如 `\x0D`），可能需要新增 `Hex1`、`Hex2` 等状态枚举。

### 3.2 Challenges

* [ ] **1. UCS2 (UTF-16) Support | UCS2 编码支持** 
    Support `\u000D` style escape sequences and correctly encode them as UTF-8 into `std::string`.
    支持 `\u000D` 格式的转义序列，并将其作为 UTF-8 编码存入 `std::string`。
* [ ] **2. UCS4 (UTF-32) Support | UCS4 编码支持** 
    Support `\U0000000D` style escape sequences and correctly encode them as UTF-8 into `std::string`.
    支持 `\U0000000D` 格式的转义序列，并将其作为 UTF-8 编码存入 `std::string`。
* [ ] **3. Universal Keys for JSONDict | 全类型键支持** 
    Allow any `JSONObject` to function as a key in `JSONDict` (requires implementing `hash` and `equal_to` traits for `JSONObject`).
    支持任意 `JSONObject` 作为 `JSONDict` 的键（需要为 `JSONObject` 实现自定义哈希 `hash` 和相等性判定 `equal_to`）。
* [ ] **4. JSON Dumper Implementation | 实现序列化器** 
    Develop a JSON dumper/writer as the functional inverse of the current parser.
    实现一个 JSON 序列化器（Dumper），作为解析器的逆向实现。
* [ ] **5. Pretty Printing | 美化输出** 
    Add an optional `isPretty` argument to the `dump()` function to output JSON with proper indentation and spacing.
    在 `dump()` 函数中添加可选参数 `isPretty`，当为 true 时，输出带有缩进和空格的美化格式。
* [ ] **6. Unquoted Keys Support | 缺省引号键支持** 
    Allow JSONDict keys to be optionally unquoted, e.g., making `{hello: "world"}` equivalent to `{"hello": "world"}`.
    支持键名省略引号，使 `{hello: "world"}` 与 `{"hello": "world"}` 等效。
* [ ] **7. YAML Extension | 扩展至 YAML** 
    Extend the current JSON parser/dumper to handle YAML (recognizing that YAML is a superset of JSON).
    现有的 JSON 解析/序列化工具扩展为 YAML 工具（利用 YAML 是 JSON 超集的特性）。
