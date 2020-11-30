new repo- git init,undoing staged changes-git reset head
restore your changes-git checkout head, switch branch-git checkout branchname
orign live git remote v
git pull-git fetch and git merge(three way merge,diff from origin and merge)
merge conflict resolve-git reset --hard head
replacing set of commits-git rebase
undo-soft(reference back to specified commit only),mixed(soft()+staging area back),hard(mixed+rolls working directory)
stash(dirty state of working directory,saves stack of unfinished changes apply later)-switch to branches but don't want commit half done work
git stash apply, git cherry-pick commit_id
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


ANDROID-

State In memory? Visible to user? In foreground?
nonexistent no no no
stopped yes no no
paused yes yes/partially* no
resumed yes yes yes
recyclerview-
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
