unit:integration:ui::70:20:10
TDD-
TDD is a process of writing tests before you write your actual code. You can apply
TDD to any sort of programming.
• Practicing TDD helps you to code quickly and with intent, document
automatically, have confidence your code is maintainable, and have better test
coverage.
• TDD follows the Red-Green-Refactor steps.
• This process always starts with writing a failing test. No matter what, you always
want to see the test fail.
• Only after you write your test and see it fail do you write your new code or change
your existing code.
You only ever write enough code to make your test pass. If there's more code you
need to write, you need another test first.
• You follow up with refactoring to make your code clean and readable.
• Learning to write good tests takes practice.

Unit test-
Unit tests verify how isolated parts of your application work.
• Using JUnit, you can write unit tests asserting results, meaning, you can compare
an expected result with the actual one.
• Every test has three phases: set up, assertion and teardown.
• In TDD, you start by writing a test. You then write the code to make the test
compile. Next you see that the test fails. Finally, you add the implementation to
the method under test to make it pass.
@After: The method will be executed after each test. You can use it to tear down
anything that you set up in @Before.
• @BeforeClass: If you annotate a method with this, it'll be executed only once
before all the tests are executed. For example, opening a file, a connection or a
database that is shared in all the tests.
• @AfterClass: To execute a method only once after all the tests are executed, use
this one. For example, closing a file, a connection or a database that is shared in all
the tests.
Architecting for testing-
Mockito-
With JUnit you can do state verification, also called black-box testing.
• With Mockito you can perform behavior verification or white-box testing.
• Using a mock of a class will let you stub methods simulate a particular situation in
a test. It'll also verify if one or more methods were called on that mock.
• Using a spy is similar to using a mock, but on real instances. You'll be able to stub a
method and verify if a method was called just like a mock, but also be able to call
the real methods of the instance.

Integration test-
 Integration tests verify the way different parts of your app work together.
• They are slower than unit tests, and should therefore only be used when you need
to test how things interact.
• When interacting with the Android framework you can rely on an Android device
or emulator, or use Robolectric.
• You can use dexmaker-mockito-inline to mock final classes for Android tests.
When running tests that require the Android framework you have two options:
1. Run them on an Android device or emulator.
2. Use Robolectric.
Robolectric is a framework that allows you to run Android-dependent tests in a unittest way. It creates a sandbox in which to run your tests; the sandbox acts like an
Android environment with Android APIs. A benefit to using Robolectric is that it is
faster, running on the JVM; using a device/emulator, however, more accurately shows
you how your code will behave when installed onto a device, having more Android
features.

Testing persistence layer-
One of the challenges that make writing persistence tests difficult is managing the
state before and after the tests run.
You can solve this problem by using an in-memory database. Room luckily provides
a way to easily create one. Add this to your test class, importing
androidx.test.platform.app.InstrumentationRegistry:
Persistence tests help keep your user's data safe.
• Statefulness can make persistence tests difficult to write.
• You can use an in-memory database to help handle stateful tests.
• You need to include both set up (@Before) and tear down (@After) with
persistence tests.
• Be careful to test your code and not the library or framework you're using.
• Sometimes you need to write "broken" code first to ensure that your tests fail.
• You can use Factories to create test data for reliable, repeatable tests.
• If the persistence library you're using doesn't have built in strategies for testing,
you may need to delete all persisted data before each test.

Testing Network layer-
There are tools that you can use to test your network layer without hitting the
network:
• MockWebserver to mock the responses from network requests
• Mockito to mock your API responses
• Faker for data creation
To keep your tests repeatable and predictable, you shouldn't make real network
calls in your tests.
• You can use MockWebServer to script request responses and verify that the correct
endpoint was called.
• How you create and maintain test data will change depending on the needs for
your app.
• You can mock the network layer with Mockito if you don't need the fine-grained
control of MockWebServer.
• By using the Faker library you can easily create random, interesting test data.
• Deciding which tools are right for the job takes time to learn through experiment,
trial, and error.

UI testing or end-to-end testing
UI testing generally verifies two things:
1. That the user sees what you expect them to see.
2. That the correct events happen when the user interacts with the screen.

Espresso-
androidTestImplementation 'androidx.test.espresso:espresso-core:
There are three main classes you need to know when working with Espresso:
ViewMatchers, ViewActions and ViewAssertions:
• ViewMatchers: Contain methods that Espresso uses to find the view on your
screen with which it needs to interact.
• ViewActions: Contain methods that tell Espresso how to automate your UI. For
example, it contains methods like click() that you can use to tell Espresso to
click on a button.
• ViewAssertions: Contain methods used to check if a view matches a specific set
of conditions.
Using dependency injec7on to set mocks- mock the repository so that you’re not hitting the network layer.
By using declareMock(), you’re overriding the provided dependency injection
Repository with a Mockito mock.
UI tests allow you to test your app end-to-end without having to manually clicktest your app.
• Using the Espresso library, you're able to write UI tests.
• You can run Android tests on a device and locally using Roboelectric.

Strategies for handling test data-
There are no magic silver bullets with test data.
• JSON data is great for quickly getting started with end-to-end tests but can
become difficult to maintain as your test suites get larger.
• Hard-coded data works well when your test suite is small, but it lacks variety in
test data as your test suite grows.
• Faker makes it easier to generate a variety of test data for object libraries.
• Tests that need to get data stores into a certain state can be expensive because you
need to insert data into the data store programmatically

Continuous Integration-
 CI helps to ensure that you are frequently integrating all developer's code to avoid
rework.
• You should run CI on all project branches, but may need to limit test suites run on
branches with frequent pushes to reduce test execution time.
• Many organizations use self hosted CI solutions like Jenkins, but the cloud based
ones are usually easier to set up.
• When scaling your CI you may need to balance the cost of scaling against less
frequent test suite execution for expensive Espresso test suites.


COROUTINES-
you can build coroutines using coroutine builders.
• The main coroutine builder is the launch function.
• Whenever you launch a coroutine, you get a Job object back.
• Jobs can be canceled or combined together using the join function.
• You can nest jobs and cancel them all at once.
• Try to make your code cooperative — check for the state of the job when doing
computational work.
• Coroutines need a scope they’ll run in.
• Posting to the UI thread in advanced applications is as easy as passing in the
Dispatchers.Main instance as the context.
• Coroutines can be postponed, using the delay function.
 Having callbacks as a means of notifying result values can be pretty ugly and
cognitive-heavy.
• Coroutines and suspendable functions remove the need for callbacks and
excessive thread allocation.
• What separates a regular function from a suspendable one is the first-class
continuation support, which the Coroutine API uses internally.
• Continuations are already present in the system, and are used to handle function
lifecycle — returning the values, jumping to statements in code, and updating the
call-stack.
• You can think of continuations as low-level callbacks, which the system calls to
when it needs to navigate through the call-stack.
• Continuations always persist a batch of information about the context in which
the function is called — the parameters passed, call site and the return type.
• There are three main ways in which the continuation can resolve - in a happy path,
returning a value the function is expected to return, throwing an exception in
case something goes bad, and blocking infinitely because of flawed business
logic.
• Utilizing the suspend modifier, and functions like launch() and
suspendCoroutine(), you can create your own API, which abstracts away the
threading used for executing code.

The async/await pattern is founded upon the idea of futures and promises, with
a slight twist in the execution of the pattern.
• Promises rely on callbacks and chained function calls to consume the value in a
stream-like syntax, which tends to be clunky and unreadable when you have a lot
of business logic.
Futures are built with tasks, which provide the value to the user, wrapped in a
container class. Once you want to receive the value, you have to block the thread
and wait for it or simply postpone getting the value until it is ready.
• Using async/await relies on suspending functions, instead of blocking threads,
which provides clean code, without the risk of blocking the user interface.
• The async/await pattern is built on two functions: async() to wrap the function
call and the resulting value in a coroutine, and await(), which suspends code
until the value is ready to be served.
• In order to migrate to the async/await pattern, you have to return the async()
result from your code, and call await() on the Deferred, from within another
coroutine. By doing so, you can remove callbacks you used to use, to consume
asynchronously provided values.
• Deferred objects are decorated by the DeferredCoroutine. The coroutine also
implements the Continuation interface, allowing it to intercept the execution
flow and pass down values to the caller.
• Once a deferred coroutine is started, it will attempt to run the block of code you
passed, storing its result internally.
• The Deferred interface also implements the Job interface, allowing you to cancel
it and check its state — the isActive and the isCompleted flags.
• You can also handle errors a deferred value might produce, by calling
getCompletionExceptionOrNull(), checking if the coroutine ended with an
exception along the way.
• By returning Deferreds from function calls, you’re able to prime multiple deferred
values, and await them all in one function call, effectively combining multiple
requests.
• Always try to create as few suspension points in your code as possible, making the
code easier to understand.
• Writing sequential, synchronous-looking code is easy using the async/await
pattern. It makes your codebase clean and requires less cognitive load to
understand the business logic.
• You can write coroutine-powered code in a bad way. Doing so, you might waste
resources or block the entire program. This is why your code should follow the
idea of structured concurrency.
Being structured means your code is connected to other CoroutineContexts and
CoroutineScopes, and carefully deals with threading and resource management.
• Always try to rely on safe CoroutineScopes, and the best way is by implementing
them yourself.
• When you implement CoroutineScope, you have to provide a CoroutineContext,
which will be used to start every coroutine.
• It’s useful to tie the custom CoroutineScope to a well-established lifecycle, like
the Android Activity.
• It’s important to write cooperative code as well, which checks the isActive state
of the parent job or scope to finish early, release resources, and avoid the potential
of a blocking thread.
All the information for coroutines is contained in a CoroutineContext and its
CoroutineContext.Elements.
• There are three main coroutine context elements: the Job, which defines the
lifecycle and can be cancelled, a CoroutineExceptionHandler, which takes care
of errors, and the ContinuationInterceptor, which handles function execution
flow and threading.
• Each of the coroutine context elements implements CoroutineContext.
• ContinuationInterceptors, which take care of the input/output of threading.
The main and background threads are provided through the Dispatchers.
• You can combine different CoroutineContexts and their Elements by using the +/
plus operator, effectively summing their elements.
• A good practice is to build a CoroutineContext provider, so you don’t depend on
explicit contexts.
• With the CoroutineContextProvider you can abstract away complex contexts,
like custom error handling, coroutine lifecycles or threading mechanisms.

Dispatchers-
One of the most important concepts in computing, when using execution
algorithms, is scheduling and context switching.
• Scheduling takes care of resource management by coordinating threading and the
lifecycle of processes.
• To communicate thread and process states in computing and task execution, the
system uses context switching and dispatching.
• Context switching helps the system store thread and process state, so that it can
switch between tasks which need execution.
• Dispatching handles which tasks get resources at which point in time.
• ContinuationInterceptors, which take care of the input/output of threading,
and the main and background threads are provided through the Dispatchers
class.
• Dispatchers can be confined and unconfined, where being confined or not relates
to using a fixed threading system.
• There are four main dispatchers: Default, IO, Main and Unconfined.
• Using the Executors class you can create new thread pools to use for your
coroutine work.

Exception-
f an exception is thrown during an asynchronous block, it is not actually thrown
immediately. Instead, it will be thrown at the time you call await on the Deferred
object that is returned.
• To ignore any exceptions, launch the parent coroutine with the async function;
however, if required to handle, the exception uses a try-catch block on the await()
call on the Deferred object returned from async coroutine builder.
• When using launch builder the exception will be stored in a Job object. To retrieve
it, you can use the invokeOnCompletion helper function.
• Add a CoroutineExceptionHandler to the parent coroutine context to catch
unhandled exceptions and handle them.
• CoroutineExceptionHandler is invoked only on exceptions that are not expected
to be handled by the user; registering it in an async coroutine builder or the like
has no effect.
• When multiple children of a coroutine throw an exception, the general rule is the
first exception wins.
• Coroutines provide a way to wrap callbacks to hide the complexity of the
asynchronous code handling away from the caller via a suspendCoroutine
suspending function, which is included in the coroutine library.

Manage Cancellation-
 When the parent coroutine is canceled, all of its children are recursively canceled,
too.
• CancellationException is not printed to the console/log by the default uncaught
exception handler.
• Using the withTimeout function, you can terminate a long-running coroutine
after a set time has elapsed.

Sequences-
Collection are eagerly evaluated; i.e., all items are operated upon completely
before passing the result to the next operator.
2. Sequence handles the collection of items in a lazy-evaluated manner; i.e., the
items in it are not evaluated until you access them.
3. Sequences are great at representing collection where the size isn’t known in
advance, like reading lines from a file.
asSequence() can be used to convert a list to a sequence.
5. It is recommended to use simple Iterables in most of the cases, the benefit of
using a sequence is only when there is a huge/infinite collection of elements
with multiple operations.
6. Generators is a special kind of function that can return values and then can be
resumed when they’re called again.
7. Using Coroutines with Sequence it is possible to implement Generators.
8. SequenceScope is defined for yielding values of a Sequence or an Iterator using
suspending functions or Coroutines.
9. SequenceScope provides yield() and yieldAll() suspending functions to enable
Generator function behavior.

Channels-. 
Channels provide the functionality for sending and receiving streams of values.
2. Channel implements both SendChannel and ReceiveChannel interfaces;
therefore, it could be used for sending and receiving streams of values.
3. A Channel can be closed. When that happens, you can’t send or receive an
element from it.
4. The send() method either adds the value to a channel or suspends the coroutine
until there is space in the channel.
5. The receive() method returns a value from a channel if it is available, or it
suspends the coroutine until some value is available otherwise.
6. The offer() method can be used as an alternative to send(). Unlike the send()
method, offer() doesn’t suspend the coroutine, it returns false instead. It
returns true in case of a successful operation.
7. poll() similarly to offer() doesn’t suspend the running, but returns null if a
channel is empty.
8. Java BlockingQueue has a similar to Kotlin Channel behavior, the main
difference is that the current thread gets blocked if the operation of inserting or
retrieving is unavailable at the moment.

Broadcast Channels-
With channels, if you have many receivers waiting to receive items from the
channel, the emitted item will be consumed by the first receiver and all other
receivers will not get the item individually.
• BroadcastChannel enables many subscribed receivers to consume all items sent
in the channel.
• ConflatedBroadcastChannel enables many subscribed receivers to consume the
most recently sent item provided the receiver consumes items slower.
• Subject from RxJava is the dual of the BroadcastChannel in behavior.
• BehaviorSubject from RxJava is the dual of ConflatedBroadcastChannel in
behavior.

Producer & Actors-
Produce-consumer pattern and the actor model are tried and tested mechanisms
for multi-threading.
• Producer-consumer relationships are one-to-many, where you can consume the
events from multiple places.
• The actor model is a way to share data in a multithread environment using a
dedicated queue.
• The actor model allows you to offload large amounts of work to many smaller
constructs.
Actors have a many-to-one relationship, since you can send events from multiple
places, but they all end up in one actor.
• Each actor can create new actors, delegating and offloading work.
• Building actors using threads can be expensive, which is where coroutines come
in handy.
• Actors can be arranged to run in sequential order, or to run in parallel.

Coroutines Flow-
Sometimes you need to build more than one value asynchronously, this is usually
done with sequences or streams.
• Sequences are lazy and cold, but blocking when you need to consume events. It's
better to use and suspend coroutines instead.
• If you build streams using Channels, then you have coroutine support and
suspendability, but they are hot by default.
 Being cold means the data isn't computed, until you start observing. As opposed
to being cold, being hot means the data is computed right away, with, or without
any observers.
• As such, streams have two sides - the producer, or observable construct, and a
consumer, or the observer construct.
• The main limitations of streams are they are blocking and use backpressure.
• Blocking happens when a stream needs to produce or consume events.
• Backpressure is when a stream is producing or consuming events too fast, and
one side has to be slowed down, to balance the stream.
• Backpressure is usually done through blocking the thread of a producer or a
consumer.
• A good stream avoids blocking, supports context switching while still allowing
for backpressure.
• The Flow API is built upon coroutines, allowing for suspending.
• Because you can suspend a consumer or a producer, you get intrinsic
backpressure support.
• Additionally, you're avoiding blocking, which is what a good stream should do.
• To create a Flow, simply call flow(), and provide a way to emit values to the
FlowCollectors which decide to process the values.
• To attach a FlowCollector to a Flow, you have to call collect(), with a lambda
in which you will consume each of the values.
• collect() is a suspending function, so it has to be within a coroutine or another
suspending function.
• You have the access to a FlowCollector from within collect(), so you can emit
values.
• Flows can be transformed and mutated by various operators like map(),
flatMap().
• You can apply manual backpressure using debounce(), delayEach() and
delayFlow().
• Switching the context of a Flow allows you to change the threads in which you
consume each piece of data, or perform each operator.
To switch context, call flowOn(context) after the operators you wish to switch
the context of.
• The Flow collects the values always in the context of the CoroutineScope it is
located in. So if you call collect() on the main thread, you'll also consume the
values there.
• Flows don't allow you to produce values concurrently. If you try to do that, an
exception will occur.
• If you do need to produce values from multiple threads, you can use
channelFlow().
• It's better to use flowOn() to switch contexts of the Flow, than to bury them down
in coroutines.
• Flows should be transparent when it comes to exceptions.
• To handle exceptions with Flows, use the catch() operator.
• catch() will intercept any uncaught exceptions, from all the operators you called
before catch() itself.

Android Concurrency Before Coroutines-
 Android is inherently asynchronous and event-driven, with strict requirements
as to which thread certain things can happen on.
• The UI thread — a.k.a., main thread — is responsible for interacting with the UI
components and is the most important thread of an Android application.
• Almost all code in an Android application will be executed on the UI thread by
default; blocking it would result in a non-responsive application state.
Thread is an independent path of execution within a program allowing for
asynchronous code execution, but it is highly complex to maintain and has limits
on usage.
• AsyncTask is a helper class that simplifies asynchronous programming between
UI thread and background threads on Android. It does not work well with complex
operations based on Android Lifecycle.
• Handler is another helper class provided by Android SDK to simplify
asynchronous programming but requires a lot of moving parts to set up and get
running.
• HandlerThread is a thread that is ready to receive a Handler because it has a
Looper and a MessageQueue built into it.
• Service is a component that is useful for performing long (or potentially long)
operations without any UI, and it runs in the main thread of its hosting process.
• IntentService is a service that runs on a separate thread and stops itself
automatically after it completes its work; however, it cannot handle multiple
requests at a time.
• Executors is a manager class that allows running many different tasks
concurrently while sharing limited CPU time, used mainly to manage thread(s)
efficiently.
• WorkManager is a fairly new API developed as part of JetPack libraries provided
by Google, which makes it easy to specify deferrable, asynchronous tasks and when
they should run.
• RxJava + RxAndroid are libraries that make it easier to implement reactive
programming principles in the Android platform.
• Coroutines make asynchronous code look synchronous and work pretty well with
the Android platform out of the box.
• Anko is a library that uses Kotlin and provides a lot of extension functions to
make our Android development easier.

Android Concurrency Using Coroutines-
CoroutineDispatcher determines what thread or threads the corresponding
coroutine uses for its execution.
2. A coroutine can switch dispatchers any time after it is started.
3. Dispatchers.Main context for Android apps, allows starting coroutines confined
to the main thread.
4. Each coroutine runs inside a defined scope.
5. A Job must be passed to CoroutineScope in order to cancel all coroutines started
in the scope.
6. Coroutines can replace callbacks for more readable and clear code
implementation.
7. Making CoroutineScope lifecycle aware helps to adhere to the lifecycle of
android components and avoid memory leaks.
8. Coroutines seamlessly integrate with WorkManager to run background jobs
efficiently.
 Debugging coroutines is pretty easy since you can name them and also log the
name of the thread they are running on.
2. To enable debugging logs in coroutines, -Dkotlinx.coroutines.debug flag
needs to be set as a JVM property.
3. By default, coroutines use the default Android policy on uncaught exception
handling if no try-catch is set up for exception handling.
4. Using CoroutineExceptionHandler, you can set up a custom handler for
exceptions generated from coroutines.
5. Dispatchers.Unconfined dispatcher is not confined to any specific thread; i.e.,
it runs on the same thread as the one on which the coroutine was launched.
6. In order to make coroutines testable, the normal dispatchers need to be replaced
by Dispatchers.Unconfined inside test methods.
7. Mockk is a Kotlin mocking library, which allows to mock coroutines as well and
tests their execution points.
8. Anko (ANdroid KOtlin) is a set of helper libraries built by the folks at JetBrain.
The Anko coroutines library is based on the standard kotlin.coroutines
library.