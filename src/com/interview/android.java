firebase,room vs realm,raywenderlich saving-data-on-android-copy firebase
DESIGN PATTERN-duck
INTERFACE quackable-quack()
xduck implement interface quackable
goose-honk(), adapter-gooseadapter implement quackable(goose, gooseadapter(goose),quack(goose.honk))
DECORATOR-quackcounter implement quackable(quackable, quack(),quackcounter(quackable))-new quackcounter(new xduck())
FACTORY-public abstract class AbstractDuckFactory {public abstract Quackable createMallardDuck()
  public class DuckFactory extends AbstractDuckFactory {
public Quackable createMallardDuck() {return new MallardDuck();}
public class CountingDuckFactory extends AbstractDuckFactory {public Quackable createMallardDuck() {
return new QuackCounter(new MallardDuck());}
AbstractDuckFactory duckFactory = new CountingDuckFactory();
Quackable mallardDuck = duckFactory.createMallardDuck();
manage,composite-public class Flock implements Quackable {
ArrayList quackers = new ArrayList();
public void add(Quackable quacker) {quackers.add(quacker);}
public void quack() {Iterator iterator = quackers.iterator()
  Flock flockOfDucks = new Flock();flockOfDucks.add(redheadDuck)
OBSERVER-individual duck behaviour
public interface QuackObservable {public void registerObserver(Observer observer);
public void notifyObservers();}
public interface Quackable extends QuackObservable {public void quack();}
public class Observable implements QuackObservable {ArrayList observers = new ArrayList();
QuackObservable duck;public Observable(QuackObservable duck) {this.duck = duck;}
public void registerObserver(Observer observer) {observers.add(observer);}
public void notifyObservers() {Iterator iterator = observers.iterator();
  public class MallardDuck implements Quackable {
Observable observable;public MallardDuck() {observable = new Observable(this);}
public void quack() {notifyObservers();}
public void registerObserver(Observer observer) {observable.registerObserver(observer);}
public void notifyObservers() {observable.notifyObservers();}
public interface Observer {public void update(QuackObservable duck);}
Quackologist quackologist = new Quackologist();
flockOfDucks.registerObserver(quackologist);

MVC pattern-
The Model consists of the data and business logic.
• The View is responsible for rendering the data in a way that is human readable on a
screen.
• The Controller is the brains behind the app and communicates with both the Model
and the View. The user interacts with the app via the Controller.
• When applying MVC to Android, the Android Activity ends up serving as both the
View and Controller, which is problematic for separation of concerns and unit
testing

MVP patern-
Unlike MVC wherein the main entry point is the Controller, in MVP the main entry
point is the View.
he Model is the data layer that handles business logic.
• The View displays the UI and informs the Presenter about user actions.
• The View extends Activity or Fragment.
• The Presenter tells the Model to update the data and tells the View to update the UI.
• The Presenter should not contain Android framework-specific classes.
• The Presenter and the View interact with each other through interfaces.

MVVM pattern-relate to MVP-seoeration of concerns
View-View displays the UI and informs the other layers about user actions.
 ViewModel exposes information to the View.
Views display the UI and inform about user actions.
The View holds a reference to your ViewModel. The ViewModel property is
defined as a lateinit var so that the compiler knows it won't be initialized until
after class initialization.
ViewModel-The ViewModel gets the information from your Data Model, applies the necessary
operations and exposes the relevant data to your Views.
It exposes backend events to the Views so they can react accordingly.
The ViewModel class is designed to store and manage UI-related data in a lifecycleaware way.
• The ViewModel class allows data to survive configuration changes, such as screen
rotations.
• LiveData is a data holder class, just like a List or a HashMap, that can be observed for
any changes within a given lifecycle.
It declares a property that will contain a LiveData list of items. The
LiveData class allows any View to observe for any changes on the list and update
the UI.
Model-The Model, also known as the DataModel, retrieves information from your backend
and makes it available to your ViewModels.
 Model retrieves information from your datasource and exposes it to the ViewModels

VIPER pattern
The View displays the User Interface.
• The Interactor performs actions related to the Entities.
• The Presenter acts as a command center to manage your Views, Interactors and
Routers.
• The Entity represents data in your app.
• The Router handles the navigation between your Views.
• VIPER is a great architecture pattern for projects that are expected to scale fast but
might be overkill for simple apps.

KOTLIN
extension function-static function defined in seperate auxiliary class-can't call private members 
from extensions
ex- standard collection-hashSetOf, arrayListOf, hashMapOf
    extension function-filter,count,flatmap,map,find,groupby-map list of list,zip-element of two list in pair,
	zipwithnext-neighbouring elements,flatten-merge list of list,flatmap-input to map then flatten
	associateby-if key unique use it else duplicates removed, associate-map on list [1,2,3,4]-{'a'+it to 10*it}
	reduce,any etc
?=nullable type-replace if expression with safe access s?.length-if null then null be result
elvis operator-s?.length?:0
s!!-throws NPE if s is null/string s-@Notnull annotation, string s?-@nullable annotation
typecast-as replace by smartcast as not needed,replace smartcast by  safecast-is means 
instanceof-(any as? string)?.touppercase() return null if throws exception
lambdas vs anonymous classes- same purpose-{params x:Int->body}
list.any({i:Int->i>0})--lambda last argument list.any(){i:Int->i>0}-if lambda only argument-list.any{i:Int->i>0}-
if argument type can be inferred from context-list.any{i->i>0}-if lambda has one argument-list.any{it>0}
member references-lambda to reference-this.maxBy{it.age}-this.maxBy(Person::age),function reference-fun isEven...--val x=::isEven
bound reference-this::isOlder
return from lambda when inside function use label-list.flatmap f@{..return@flatmap or @f}
lateinit isinitialized-can't be val,nullable,primitive type
by lazy function takes lambda as arg-its value computed only once
default modifiers-public and final, open can be overridden
public-class member & top level declaration-visible everywhere
internal- ...........................-visible in the module
protected-visible in subclasses
private-visible in class and file
coursera android specialization testing
kotlin data modifier-=== reference,== content
kotlin inner and nested classes-inner classes stores reference to outer class-static nested class by default
doesn't store reference of outer class,if need reference explicit use inner modifier inner class C{this@A}
sealed modifier restricts class hierarchy-all subclasses must be located in the same file-interface 
changed to sealed class
object declaration-singleton, object expression replaces anonymous classes,companion object inside 
class can implement interface
constants- for primitive type use const, for reference type use @JvmField elimintes accessors
@Jvmstatic-call getter,@Jvmfield-calls field, const-resulting bytecode replaced with actual value
Null-
 null represents the absence of a value.
• Non-null variables and constants must always have a non-null value.
• Nullable variables and constants are like boxes that can contain a value or be
empty (null).
• To work with the value inside a nullable, you must typically first check that the
value is not null.
• The safest ways to work with a nullable’s value is by using safe calls or the Elvis
operator. Use not-null assertions only when appropriate, as they could produce a
runtime error.
Collection-
 Arrays are ordered collections of values of the same type.
• There are special classes such as IntArray created as arrays of Java primitive
types.
• Lists are similar to arrays but have the additional feature of being dynamicallysized.
• You can add, remove, update, and insert elements into mutable lists.
• Use indexing or one of many methods to access and update elements.
• Be wary of accessing an index that’s out of bounds.
• You can iterate over the elements of an array or list using a for loop or using
forEach.
• You can check for elements in an array or list using in.
• Special consideration should be given when working with nullable lists and lists
with nullable elements.
Lambdas-
• Lambdas are functions without names. They can be assigned to variables and
passed as arguments to functions.
• Lambdas have shorthand syntax that makes them a lot easier to use than other
functions.
• A lambda can capture the variables and constants from its surrounding context.
• A lambda can be used to direct how a collection is sorted.
• There exists a handy set of functions on collections which can be used to iterate
over the collection and transform the collection. Transforms include mapping each
element to a new value, filtering out certain values, and folding or reducing the
collection down to a single value.
Classes-
Classes are a named type that can have properties and methods.
• Classes use references that are shared on assignment.
• Class instances are called objects.
• Objects are mutable.
• Mutability introduces state, which adds complexity when managing your objects.
• Data classes allow you to create simple model objects that avoid a lot of
boilerplate for comparing, printing, and copying objects.
• Destructuring declarations allow you to easily extract multiple properties of
data class objects.
• Class inheritance is one of the most important features of classes and enables
polymorphism.
• Subclassing is a powerful tool, but it’s good to know when to subclass. Subclass
when you want to extend an object and could benefit from an "is-a" relationship
between subclass and superclass, but be mindful of the inherited state and deep
class hierarchies.
• The open keyword is used to allow inheritance from classes and also to allow
methods to be overridden in subclasses.
• Sealed classes allow you to create a strictly defined class hierarchy that is similar
to an enum class but that allow multiple instances of each subtype to be created
and hold state.
• Secondary constructors allow you to define additional constructors that take
additional parameters than the primary constructor and take different actions
with those parameters.
• Nested classes allow you to namespace one class within another.
• Inner classes are nested classes that also have access to the other members of the
outer class.
• Visibility modifiers allow you to control where class members and top-level
declarations can be seen within your code and projects.
Sealed vs enum classes-
They are abstract. This means that you can’t instantiate an instance of the sealed
class directly, only one of the declared subclasses.
• Related to that requirement, sealed classes can have abstract members, which
must be implemented by all subclasses of the sealed class.
• Unlike enum classes, where each case is a single instance of the class, you can have
multiple instances of a subclass of a sealed class.
• You can’t make direct subclasses of a sealed class outside of the file where it’s
declared, and the constructors of sealed classes are always private.
• You can create indirect subclasses (such as inheriting from one of the subclasses of
your sealed class) outside the file where they’re declared, but because of the
restrictions above, this usually doesn’t end up working very well.
Enum classes are a powerful tool for handling situations where a piece of data will
(or at least should) be one of a defined set of pre-existing values. Enum classes
come with a number of tools for free, such as getting a list of all the declared cases,
and the ability to access the order and names of the cases.
• Sealed classes are a powerful tool for handling situations where a piece of data will
(or at least should) be one of a defined set of pre existing types.
• Both enum classes and sealed classes let you take advantage of Kotlin’s powerful
when expression to clearly outline how you want to handle various situations.
• Enum classes are particularly useful for creating, updating, and cleaning
information about the current state in a state machines.

Objects-
• The singleton pattern is used when you want only one instance of a type to be
created in your app.
• The object keyword is unique to Kotlin compared with similar languages, and it
gives you a built-in way to make singletons with named objects. It also lets you
make anonymous objects, the Kotlin version of Java anonymous classes.
• A class companion object gives you the Kotlin equivalent of Java static members.
• Anonymous objects — or object expressions — let you create unnamed
instances of interfaces and to override class behavior without subclassing.
• Using Show Kotlin Bytecode and decompiling in IntelliJ IDEA is an informative
way to understand what the Kotlin compiler is doing.
Properties-
Properties are variables and constants that are part of a named type.
• Default values can be used to assign a value to a property within the class
definition.
• Property initializers and the init block are used to ensure that the properties of
an object are initialized when the object is created.
• Custom accessors are used to execute custom code when a property is accessed or
set.
• The companion object holds properties that are universal to all instances of a
particular class.
• Delegated properties are used when you want to observe, limit or lazily create a
property. You’ll want to use lazy properties when a property’s initial value is
computationally intensive or when you won’t know the initial value of a property
until after you’ve initialized the object.
• lateinit can be used to defer setting the value of a property reference until after
the instance is created.
• Extension properties allow you to add properties to a class outside of the class
definition, for example, if you’re using a class from a library.
Methods-
• Methods are behaviors that extend the functionality of a class.
• A typical method is a function defined inside of a class or object.
• A method can access the value of an instance by using the keyword this.
• Companion object methods add behavior to a class instead of the instances of
that class. To define a companion object method, you add a function in the class
companion object block.
• You can augment an existing class definition and add methods to it using
extension methods.

Interfaces-
Interfaces define a contract that classes, objects, and other custom types can
implement.
• By implementing an interface, a type is required to conform to the interface by
implementing all methods and properties of the interface.
• A type can implement any number of interfaces, which allows for a quasi-multiple
inheritance not permitted through subclassing.
• The Kotlin standard library uses interfaces extensively. You can use many of them,
such as Comparable, on your own types.
Generics-
 Generics allow you to create classes or interfaces that operate on a type that is not
known when your code for that class or interface is written.
• Generic programming can allow you to centralize pieces of functionality in a
highly reusable and easily debuggable fashion.
• Type erasure means that, within a class or interface that takes a generic type, you
won't have any information about that type at compile time unless you annotate
the type with reified and inline the function.
• Allowing only in or out variance of a generic type allows you to restrict whether a
generic type can be passed in to extensions or be returned from subclasses or
other functions on a particular generic interface or class. This, in turn, allows both
you and the compiler to make assumptions about how generic types relate to each
other.

Kotlin Interoperability-
Kotlin was designed from the beginning to be compatible with the JVM, and Kotlin
bytecode can run anywhere that Java bytecode runs.
• You can intermix Kotlin and Java code within one project.
• It's possible to add Kotlin extension functions to classes written in Java, and also
to call Kotlin free functions from Java code.
• Annotations like @JvmOverloads and @JvmStatic help you integrate your Java
and Kotlin code.

Exceptions-
Exceptions are the events that happen when something goes wrong in your
program.
• Extend the Exception class or its subclasses to create custom exceptions.
• Throw an exception using the throw keyword.
• Do not catch the base class Exception, use the most specific exception class you
can.
• Create custom exceptions for uncommon cases to differentiate them.
• When handling exceptions, place the code that should be executed whether an
exception occurs or not in the finally block.
• All exceptions in Kotlin are unchecked.
• Don't ignore exceptions.
• Try-catch is an expression.

Functional programming-
Functional programming uses first-class functions, which can be passed as
arguments, returned or assigned to variables.
• A higher-order function is a function that receives another function as a
parameter and/or returns one.
• A lambda is a function literal defined in curly brackets, and can be invoked,
passed to a function, returned or assigned to a variable.
When you create a lambda, an implicit class is created that implements a
FunctionN interface, where N is number of parameters that the lambda receives.
• Kotlin lambdas act as closures, with access variables defined in the outer scope of
the lambda.
• Extension functions implicitly receive an instance of the extended class as the first
parameter.
• Lambdas with receivers are similar to extension functions.
• Mark a lambda that shouldn't support a non-local return with the crossinline
keyword.
• Use the tailrec keyword to optimize tail-recursive functions.
• Use the inline keyword to replace a function invocation with its body.
• If a function is a member function or extension function, and it receives only one
argument, you can mark it with an infix keyword and call it without the dot
operator or parentheses.
• Use sequences to create lazily evaluted collections.

Coroutines-
The asynchronous approach to programming focuses on allowing you to execute
several operations at the same time.
• Threads are used when you don't need a lot of them to perform the necessary
tasks.
• Coroutines are like "lightweight threads", since they don't require as much
memory resources and they're not based on OS level threads like Java threads.
• A large number of coroutines could be executed on a single thread without
blocking it.
• Each coroutine is bound to some CoroutineContext.
• CoroutineContext is responsible for many important parts of a coroutine such as
its Job, Dispatcher and CoroutineExceptionHandler.
• Use coroutines builders (runBlocking(), withContext(), launch(), async()) to
create and launch coroutines.
You can decide when to launch your coroutine using CoroutineStart.
• Use dispatchers to define the threads for your coroutine execution.
• Coroutines are based on the concept of a state machine, with each state referring
to a suspension point. It doesn't require extra time or resources to switch between
coroutines and to restore their state.

Scripting-
As a scripting language, Kotlin gives you the static-typing lacking in other
scripting languages like Python and Ruby.
• Kotlin comes with a Read-Evaluate-Print Loop or REPL that can be used to
investigate Kotlin code in an interactive manner.
• Kotlin scripts end with the extension .kts, as opposed to normal Kotlin code that
ends with .kt.
• You can use IntelliJ IDEA as a script editor, and then either run your scripts within
the IDE or from a command line shell on your OS.
• Kotlin scripts run inside a hidden main() function and can access args passed into
the script at the command line. Scripts can also import Kotlin and Java libraries to
access their features.

Native-
 Kotlin/Native is used to compile Kotlin code to native binaries that can be run
without virtual machines and Kotlin runtimes.
• The name of the Kotlin/Native compiler is Konan. The command to run Konan is
konanc.
• Kotlin/Native leverages the LLVM compiler to produce native binaries. Konan acts
as a front-end for LLVM, producing an Intermediate Representation for LLVM.
• The Kotlin/Native compiler can be installed from it's GitHub page at https://
github.com/kotlin-native.
• When installing the Kotlin/Native compiler, be sure to distinguish it from the
kotlin-jvm compiler used to create Java bytecode.
• Kotlin/Native code starts with a main function, similar to other C-like languages.
• The Kotlin standard library is statically linked to Kotlin/Native executables.

Multiplaform-
Kotlin Multiplatform is a new and fast-growing approach to cross-platform app
development.
• KMP lets you share Kotlin code between iOS, Android, web, server, and more.
• There are a number of advantages to KMP, including developer familiarity with
Kotlin, native performance, native UI code, and the consolidation of your app
business logic into a shared module across all platforms.
• You use the expect and actual keywords to create a common interface within the
shared code that relies on concrete implementations on different platforms as
needed.


ANDROID-

State In memory? Visible to user? In foreground?
nonexistent no no no
stopped yes no no
paused yes yes/partially* no
resumed yes yes yes

Activity-
activity-visible-onstart-onstop,foreground-onresume-onpause
The activity gets launched, and the
onCreate() method runs.
Any activity initialization code in the
onCreate() method runs. At this point, the
activity isn’t yet visible, as no call to onStart()
has been made.
The onStart() method runs. It gets
called when the activity is about to
become visible.
After the onStart() method has run, the user
can see the activity on the screen
The onResume() method runs. It gets
called when the activity is about to move
into the foreground.
After the onResume() method has run, the
activity has the focus and the user can interact
with it.
The onPause() method runs when the
activity stops being in the foreground.
After the onPause() method has run, the
activity is still visible but doesn’t have the focus.
If the activity moves into the
foreground again, the onResume()
method gets called.
The activity may go through this cycle many
times if the activity repeatedly loses and then
regains the focus.
The onStop() method runs when the
activity stops being visible to the user.
After the onStop() method has run, the activity
is no longer visible.
If the activity becomes visible to the
user again, the onRestart() method gets
called followed by onStart().
The activity may go through this cycle many
times if the activity repeatedly becomes invisible
and then visible again
Finally, the activity is destroyed.
The onStop() method will get called before
onDestroy().

FRAGMENT-
onAttach(Context)
This happens when the fragment is associated with a
context, in this case an activity.
onCreate(Bundle)
This is very similar to the activity’s onCreate() method.
It can be used to do the initial setup of the fragment.
onCreateView(LayoutInflater, ViewGroup, Bundle)
Fragments use a layout inflater to create their view at this
stage.
onActivityCreated(Bundle)
Called when the onCreate() method of the activity
has completed.
onStart()
Called when the fragment is about to become visible.
onResume()
Called when the fragment is visible and actively running.
onPause()
Called when the fragment is no longer interacting with
the user.
onStop()
Called when the fragment is no longer visible to the user.
onDestroyView()
Gives the fragment the chance to clear away any
resources that were associated with its view.
onDestroy()
In this method, the fragment can clear away any other
resources it created.
onDetach()
Called when the fragment finally loses contact with the
activity.

RecyclerView-
An OnScrollListener can be added to a RecyclerView to receive
 messages when a scrolling event has occurred on that RecyclerView.
The adapter is responsible for:
• creating the necessary ViewHolders when asked
• binding ViewHolders to data from the model layer when asked
The recycler view is responsible for:
• asking the adapter to create a new ViewHolder
• asking the adapter to bind a ViewHolder to the item from the backing data at a given position
1.The RecyclerView asks the Adapter how many items it has and for an item or a ViewHolder at a given position.
2. The Adapter reaches into a pool of created ViewHolder it has.
3. Either a ViewHolder is returned or a new one is created.
4. The Adapter binds the ViewHolder to a data item at the given position.
5. The ViewHolder is returned to the RecyclerView for display.
Adapter.onCreateViewHolder(…) is responsible for creating a view to display, wrapping the view
in a view holder, and returning the result. In this case, you inflate list_item_view.xml and pass the
resulting view to a new instance of CrimeHolder.
Adapter.onBindViewHolder(holder: CrimeHolder, position: Int) is responsible for populating
a given holder with the crime from a given position. In this case, you get the crime from the crime
list at the requested position. You then use the title and data from that crime to set the text in the
corresponding text views.
When the recycler view needs to know how many items are in the data set backing it (such as when
the recycler view first spins up), it will ask its adapter by calling Adapter.getItemCount(). Here,
getItemCount() returns the number of items in the list of crimes to answer the recycler view’s request
The RecyclerView calls the adapter’s onCreateViewHolder(ViewGroup, Int) function to create a
new ViewHolder, along with its juicy payload: a View to display. The ViewHolder (and its itemView)
that the adapter creates and hands back to the RecyclerView has not yet been populated with data from
a specific item in the data set.
Next, the RecyclerView calls onBindViewHolder(ViewHolder, Int), passing a ViewHolder into this
function along with the position. The adapter will look up the model data for that position and bind it
to the ViewHolder’s View. To bind it, the adapter fills in the View to reflect the data in the model object.
After this process is complete, RecyclerView will place a list item on the screen.

RXJAVA,DAGGER,VIEWMODEL,RETROFIT,MVVM,GLIDE,DATA BINDING,ROOM,PAGING 3
GLIDE-prevent Glide from making a network call based on a particular state
RETROFIT-
Model class which is used as a JSON model
Interfaces that define the possible HTTP operations
Retrofit.Builder class - Instance which uses the interface and the Builder API to allow defining
the URL end point for the HTTP operations.
A Retrofit class: This is where you’ll create a Retrofit instance and define the base URL that your app will
use for all of its HTTP requests.
An Interface that defines the HTTP operations: This where you’ll describe each Retrofit request
that you want to make, using special Retrofit annotations that contain details about the parameters and the request method.
A POJO: This is a data model class that ensures the server’s response gets mapped automatically,
so you don’t have to perform any manual parsing.
A synchronous or asynchronous network request: Once you’ve crafted your network request, you’ll need
to execute it, and specify how your application should handle the response — whether that’s a success or a failure.
DATA BINDING-
Data binding lets you display the values of variables or properties inside XML
layouts. Data binding is not enabled by default; you need to activate it in the app-level
build.gradle.
• Two-way data binding lets you set values and react to changes at the same time.
• The two-way binding syntax @={}, lets you update the appropriate values in the
ObservableFields.
• The one-way binding syntax @{}, lets you display a certain property from the
viewmodel in the assignment expression.

ROOM-
Entities in Room represent tables in your database.
• DAO stands for Data Access Object.
• The Repository class handles the interaction with your Room database and other
backend endpoints.
• The ViewModel communicates the data coming from your repository to your views
and has the advantage of surviving configuration changes since it's lifecycleaware.
• LiveData is a data holder class that can hold information and be observed for change
The @Ignore annotation tells Room to ignore a certain property from your class so
it does not get converted into a column in the database.
• The @Embedded annotation can be used on an entity's field to tell Room that the
properties on the annotated object should be represented as columns on the same
entity. This way you can organize your data clearly while writing the same SQL.
One to one relationship: When one entity can only be related to one and only
one instance of the other entity.
• One to many relationship: When one entity can be related to many instances of
another entity.
• Many to many relationship: When many instances of an entity can also be
related to many instance of another entity
Migration(startVersion, endVersion) is the base class for a database
migration. It can move between any two migrations defined by the startVersion
and endVersion parameters.
• fallbackToDestructiveMigration() tells Room to destructively recreate tables
if you haven't specified a migration.
DAO- are objects that provide access to your app's data by abstracting most of the
complexity behind querying and updating your database.
• In Room, DAOs can be defined as interfaces or abstract classes.
• The @Insert annotation is a marker, which allows you to automatically create an
INSERT INTO query.
• @Insert can take an OnConflictStrategy parameter, that allows you to specify
what happens in case there is a conflict when creating a new database entry.
• @Query allows you to perform any kind of queries in your database. You can also
use autocomplete, to easily connect to entities and their property definitions.
• @Transaction tells Room that the following SQL statements should be executed
in a single transaction.
• @Delete allows you to automatically create DELETE FROM queries, but it requires
a parameter to be removed; e.g., a Question object.
• @Update updates a record in the database if it already exists, or omits the
changes, if it doesn't, leaving the database unchanged.
• Writing tests with Espresso is a good way to see if your database code works
properly.
• You can run Espresso tests, without manually going through the app, in less
than a few seconds.
• Inserting the data and reading from the databse in Espresso is safe, because you
can work with an in-memory version of the database.
• In-memory databases clear up after tests end, so there's no need to do extra
cleanup, other than to close() the database, to avoid leaks.

LIVEDATA- LiveData is a data holder class, like a List, that can be observed for changes by an
Observer.
• LiveData is lifecycle-aware, meaning it can observe the lifecycle of Android
components like the Activity or Fragment. It will only keep updating observers if
its component is still active.
• The ViewModel is part of the Google's architecture components and it's
specifically designed to manage data related to your user interface.
• A Repository helps you separate concerns to have a single entry point for your
app's data.
• You can combine LiveDatas and add different sources, to take action if
something changes.

PAGING 3-
PagedListAdapter has been renamed to PagingDataAdapter. The implementation is still the same.
DataSource has changed to PagingSource. ItemKeyedDataSource, PageKeyedDataSource and PositionalDataSource
 have been merged to one PagingSource class. And you no longer have to override loadInitial, loadAfter
 and loadBefore. Instead, you have one load method, which is a suspend function. This means you don't need
 to use callbacks as before.
LivePagedListBuilder has changed to Pager. DataSource.Factory is no longer required when creating a Pager instance.
PagedList.Config has been renamed to PagingConfig, but its implementation remains the same.


observable emit events, oberserver react to events, viewmodel has an obeserver object
interface(@get http endpoint,getfunc-return observable variable), service(retrofit-GSON(convert json into model objects),
AdapterFactory(Rxjava-convert date into observable variable),baseurl)
subscribeon(background thread),observeon(main thread),subscribewith(disposablesingleobserver)
@inject-which variables to inject, @module-defines how to create objects we want to inject
@component-which links two together
seperate creation/composition of the service api and use/function of the service api
composition/creation- apimodule_@module(provideorcreateapi_@provides),
interface apicomponent(@component(module=apimodule))- fun injecct(servicefunctionapi or use),
servicefunctionclass(@Inject lateinit variale functionapi,daggerapicomponent.create.inject(this))

RXJAVA-
Everything is a sequence in RxJava, and the primary sequence type is Observable,create by .just()
• Observables start emitting when they are subscribed to.
• You must dispose of subscriptions when done with them, and you’ll often use a
CompositeDisposable to do so.
• Single, Completable and Maybe are specialized observable types that are handy
in certain situations.

• Subjects are Observables that are also observers.
• You can send events over subjects by using onNext, onError and onComplete.
• PublishSubject is used when you only want to receive events that occur after
you’ve subscribed.
• BehaviorSubject will relay the latest event that has occurred when you subscribe,
including an optional initial value.
• ReplaySubject will buffer a configurable number of events that get replayed to
new subscribers. You must watch out for buffering too much data in a replay
subject.
• AsyncSubject only sends subscribers the most recent next event upon a complete
event occurring.
• The RxRelay library can be used with relays in place of subjects, to prevent
accidental complete and error events to be sent.

APP PERFORMANCE-
jetpack benchmark lib @rule,tests in CI without Gradle, or locally if you're using a different build system
Benchmark outputs a JSON file with results and metadata from a test run. This JSON file is written to
external storage on the device. You must to pull the file from the device with each run.
strictmode-long running operations,traceview,gpu profiling,android profiler

CUSTOM VIEW-
How to make basic shapes using Custom Views
How to add custom attributes to your Custom Views
How to make shape manipulations using Custom Views (increase/decrease shape size, change shape colour using functions)
How to add Accessibility Hooks onto your Custom Views
“MyCustomView”, and extend in by View class.
At this point, android Studio will prompt you to an error to create constructor(s) matching super.
On clicking the prompt, you should select all the options for the constructor.
4. Next, you create a new function void init(@Nullable AttributeSet set) with blank body and
make all the constructors access this function by calling init(attrs) on all constructors
 (except you have to pass null in the first constructor) ( to be used later).
 5. Override the onDraw(Canvas canvas) in this class. In this function you have to:
 Create a new Paint object and assign a colour to it, Create a Rect object and assign left,
 right, top, bottom coordinates to it ( Please note that a shape in canvas has positive coordinates
from top to bottom and left to right), then call canvas.
Accessibility hooks-
1.Implementing accessibility API methods like sendAccessibilityEvent(), sendAccessibilityEventUnchecked(),
 dispatchPopulateAccessibilityEvent()
 2.sendAccessibilityEvent():override onTouchEvent(MotionEvent event), then apply switch case on event.getAction(), then at
 case MotionEvent.ACTION_DOWN add sendAccessibilityEvent(AccessibilityEvent.TYPE_VIEW_CLICKED).
 3. sendAccessibilityEventUnchecked(): This method performs the sending of accessibility events
 same as sendAccessibilityEvent(), except the fact that action taken
  on these “sent” accessibility event occurs regardless of the fact if or if not
   accessibility is enabled in the system settings.
  4. dispatchPopulateAccessibilityEvent(): The default implementation of this method calls
   onPopulateAccessibilityEvent() for this view and then the dispatchPopulateAccessibilityEvent()
   method for each child of this view.
  5. onPopulateAccessibilityEvent(): This method sets the spoken text prompt of the
  AccessibilityEvent for your view. This method is also called if the view is a child of a view
   which generates an accessibility event
  6. onInitializeAccessibilityEvent(): The system calls this method to obtain additional
  information about the state of the view
  7. onInitializeAccessibilityNodeInfo(): This method provides accessibility services
  with information about the state of the view, particularly when your view extends the bounds of a simple view
  8. onRequestSendAccessibilityEvent(): The system calls this method when a child of your view has generated an 
  AccessibilityEvent. This step allows the parent view to amend the accessibility event with additional
  information. You should implement this method only if your custom view can have child views
  and if the parent view can provide context information to the accessibility event that would be
   useful to accessibility services.

GIT
new repo- git init,undoing staged changes-git reset head
restore your changes-git checkout head, switch branch-git checkout branchname
orign live git remote v
git pull-git fetch and git merge(three way merge,diff from origin and merge)
merge conflict resolve-git reset --hard head
replacing set of commits-git rebase
undo-soft(reference back to specified commit only),mixed(soft()+staging area back),hard(mixed+rolls working directory)
stash(dirty state of working directory,saves stack of unfinished changes apply later)-switch to branches but don't want commit half done work
git stash apply, git cherry-pick commit_id

DOCKER
docker compose	up	starts	up	all	the	containers.
• docker compose	ps checks	the	status	of	the	containers	managed	by	docker compose.
• docker compose	logs outputs	colored	and	aggregated	logs	for	the	compose-managed
containers.
• docker compose	logs with	dash	f	option	outputs	appended	log	when	the	log	grows.
• docker compose	logs with	the	container	name	in	the	end	outputs	the	logs	of	a	specific
container.
• docker compose	stop stops	all	the	running	containers	without	removing	them.
• docker compose	rm removes	all	the	containers.
• docker compose	build rebuilds	all	the	images.
bridge-default ,none-isolated,host
dockermachine -multi host environment via vm
docker start relate to docker compose in docker swarm