here is what I learned

from the basement code, mostly deep dive into the `decorator`
from boson-agent there are 3 main decorators; `@tool`, `@hook`, `@rule`

and those decorator help to register but also easy to wrap up into usable format. 

decorator itself can be used as a type of looger too, but the most important functionality is `hook`
`hook` can modify the middle of the code or change the state whenever I need to check some points. 

the mofrmat of decorator is a nested fuctions. the reaoson of nested function is to block decorator call the input function direcrtly. 
so the overal shape is like

```
def count_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{end-start}")

    return wrapper

@count_time
def sum(a, b):
    return a+b
```

and decorator itself can take variable too. for example, if I wnat to repeat some function N times, 

```
def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def sum(a, b):
    return a+b
```


I think this is the very basic knowledge to understand the design choice of `boson-agent`

my question was why most of the function inside of single lopp included `async` instead of `sync`
the reason over here is SINGLE GATEWAY manage all the different sessions. so that to manage the multiple session at the same time without waiting for one session to finish, it uses `async`

except that one, there is so many design choice like meta-tool, permission, skill, ...
I think we need to check the purpose of meta-tool here. 
there is clear trade-off from meta-tool. it can save tokens and let agent to use multiple tools that alive in context, but cannt effectively limit of tool usage (maybe can control thorugh the prompt but it cannot limit systemetically calling the tool itself)


from gateway, it is also based on `async` and also.  handle some amount of interruption, but didn't dig it that much. instead focusing on how overal gateway works. 
manage the session by session, depends on the result of `rules` from gateway, it can modify the input to Agent (I mean user message) like adding hook, change stage, pre-load tool/skills. 
and `rules`, tool, skills are based on the folder system, and the code itself includes automatically search all decorator and load it in to session. 

also from tool calling side, once flag 'meta-tool' then input is change. before that it support native tool callings. 
